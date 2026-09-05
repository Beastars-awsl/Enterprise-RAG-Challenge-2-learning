"""
02/03 阶段（离线）：把 01 的「逐块 JSON」渲染成按页的、可读且可切块的 markdown 风格文本。

职责定位:
    PageTextPreparation 是文档结构到「检索友好文本」的转换器 —— 也是全部召回效果的
    地基之一：它决定哪块文字被保留、块与块之间以什么层级/空行组织（markdown 标题是
    切分器与 LLM 理解章节边界的主要线索）。

数据流位置:
    输入: debug_data/01_parsed_reports/*.json（JsonReportProcessor 产物）；
    输出: process_reports -> debug_data/02_merged_reports(_ser_tab)/*.json
          （content.pages: [{page, text}]，供 text_splitter 切块）；
          export_to_markdown -> debug_data/03_reports_markdown(_ser_tab)/*.md
          （供人工审查 / full_context 问答配置）。

关键规则（对应下游假设，改动需谨慎）:
    - 按块类型做白名单过滤：页眉页脚、图片块整块丢弃；
    - 文本清洗: 把 docling/OCR 残留的排版控制记号（形如 /two.period、/three.tnum 的
      LaTeX 风格命令、glyph<..> 标签、/X.cap）替换回可读字符，并计数汇报；
    - 结构还原: 表格/列表与其前导冒号段落、后续脚注聚成一组渲染（避免表格孤悬、
      脚注与正文失联）；页内前 3 块启发式决定标题层级（H1 报告标题 vs H2 章节标题）；
    - use_serialized_tables=True 时表格输出为 markdown + 序列化描述叠加文本
      （缺 serialized 字段自动回退纯 markdown）。

核心依赖与副作用:
    - 纯 CPU 文本处理，无 I/O 阻塞除文件读写；幂等（输入不修改，只产出新文件）；
    - 依赖 01 JSON 的块类型标签（type 枚举: page_header/page_footer/section_header/
      paragraph/text/table/list_item/footnote/caption/formula/checkbox_*/picture）——
      遇到未知类型直接 raise（fail-fast 暴露 schema 漂移）。
"""
import re
from typing import List, Tuple
from pathlib import Path
import json

class PageTextPreparation:
    """
    01 逐块 JSON -> 按页 markdown 文本的转换器（页文本 = 检索与切块的基本单元）。

    唯一可复用的核心组件是排版规则引擎 _apply_formatting_rules；单报告处理入口为
    process_report，全目录入口为 process_reports。

    线程安全: process_report 会把当前报告缓存到 self.report_data（供按 id 取表），
    因此同一实例不可并发处理多份报告；多报告并行应各建实例。


    Notes:
        use_serialized_tables 与 serialized_tables_instead_of_markdown 组合:
        (False, *)      -> 纯 markdown 表格；
        (True, False)   -> markdown + 序列化描述（默认 ser_tab 形态，二者都进文本）；
        (True, True)    -> 只用序列化描述，丢弃 markdown（检索语料瘦身形态）。
    """

    def __init__(self, use_serialized_tables: bool = False, serialized_tables_instead_of_markdown: bool = False):
        """Initialize with option to add serialized tables to markdown ones."""
        self.use_serialized_tables = use_serialized_tables
        self.serialized_tables_instead_of_markdown = serialized_tables_instead_of_markdown

    def process_reports(
        self, 
        reports_dir: Path = None, 
        reports_paths: List[Path] = None, 
        output_dir: Path = None
    ):
        """
        Process reports from a directory or list of paths, returning a list of processed reports 
        and saving them to an output directory if specified.
        """
        all_reports = []
        
        if reports_dir:
            reports_paths = list(reports_dir.glob('*.json'))
        
        for report_path in reports_paths:
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            
            full_report_text = self.process_report(report_data)
            report = {"metainfo": report_data['metainfo'], "content": full_report_text}
            all_reports.append(report)
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / report_path.name, 'w', encoding='utf-8') as file:
                    json.dump(report, file, indent=2, ensure_ascii=False)
        
        return all_reports
        
    def process_report(self, report_data):
        """处理单份报告：逐页 清洗+排版 -> 页文本列表，并汇报文本修正数量。

        Args:
            report_data: 01 解析 JSON（content 为页块列表；tables 供按 id 渲染表格）

        Returns:
            {"chunks": None, "pages": [{"page": 页码, "text": markdown 文本}, ...]}
            —— 02 文件与 text_splitter 的输入契约（chunks 占位保留，由切块阶段填充）。

        副作用: 发生清洗修正时向 stdout 打印数量与前 30 条明细（语料质量巡检用）。
        """
        self.report_data = report_data
        processed_pages = []
        total_corrections = 0
        corrections_list = []

        for page_content in self.report_data["content"]:
            page_number = page_content["page"]
            page_text = self.prepare_page_text(page_number)
            cleaned_text, corrections_count, corrections = self._clean_text(page_text)
            total_corrections += corrections_count
            corrections_list.extend(corrections)
            page_data = {
                "page": page_number,
                "text": cleaned_text
            }
            processed_pages.append(page_data)
        
        if total_corrections > 0:
            print(
                f"Fixed {total_corrections} occurrences in the file "
                f"{self.report_data['metainfo']['sha1_name']}"
            )
            print(corrections_list[:30])
        
        processed_report = {
            "chunks": None,
            "pages": processed_pages
        }
        
        return processed_report

    def prepare_page_text(self, page_number):
        """Main method to process page blocks and return assembled string."""
        page_data = self._get_page_data(page_number)
        if not page_data or "content" not in page_data:
            return ""

        blocks = page_data["content"]

        filtered_blocks = self._filter_blocks(blocks)
        final_blocks = self._apply_formatting_rules(filtered_blocks)

        if final_blocks:
            final_blocks[0] = final_blocks[0].lstrip()
            final_blocks[-1] = final_blocks[-1].rstrip()

        return "\n".join(final_blocks)

    def _get_page_data(self, page_number):
        """Returns page dict for given page number, or None if not found."""
        all_pages = self.report_data.get("content", [])
        for page in all_pages:
            if page.get("page") == page_number:
                return page
        return None

    def _filter_blocks(self, blocks):
        """Remove blocks of ignored types."""
        ignored_types = {"page_footer", "picture"}
        filtered_blocks = []
        for block in blocks:
            block_type = block.get("type")
            if block_type in ignored_types:
                continue
            filtered_blocks.append(block)
        return filtered_blocks
    
    def _clean_text(self, text):
        """清洗 docling 排版记号并统计修正次数（修正明细由调用方打印前 30 条）。

        要处理的记号族:
            1) 「斜杠命令」/zero.pl.tnum 一类 —— 数学/表格识别把 0、.、逗号等符号
               转成了命令式记号（含 .pl/.tnum/.sups 等词形/序数后缀），映射回字面量；
            2) glyph<...> 残片 —— 不可读字形引用，直接删除（修正为空串）；
            3) /X.cap —— 大写转义（cap=capital），还原为对应大写字母。
        修正数 = 三族模式的总命中数（先对原文本计数再做替换，避免替换后文本变化影响统计）。

        Returns:
            (cleaned_text, occurrences_amount, corrections)
        """
        command_mapping = {
            'zero': '0',
            'one': '1', 
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'period': '.',
            'comma': ',',
            'colon': ":",
            'hyphen': "-",
            'percent': '%',
            'dollar': '$',
            'space': ' ',
            'plus': '+',
            'minus': '-',
            'slash': '/',
            'asterisk': '*',
            'lparen': '(',
            'rparen': ')',
            'parenright': ')',
            'parenleft': '(',
            'wedge.1_E': '',
        }

        recognized_commands = "|".join(command_mapping.keys())
        slash_command_pattern = rf"/({recognized_commands})(\.pl\.tnum|\.tnum\.pl|\.pl|\.tnum|\.case|\.sups)"

        occurrences_amount = len(re.findall(slash_command_pattern, text))
        occurrences_amount += len(re.findall(r'glyph<[^>]*>', text))
        occurrences_amount += len(re.findall(r'/([A-Z])\.cap', text))

        corrections = []

        def replace_command(match):
            base_command = match.group(1)
            replacement = command_mapping.get(base_command)
            if replacement is not None:
                corrections.append((match.group(0), replacement))
            return replacement if replacement is not None else match.group(0)

        def replace_glyph(match):
            corrections.append((match.group(0), ''))
            return ''

        def replace_cap(match):
            original = match.group(0)
            replacement = match.group(1)
            corrections.append((original, replacement))
            return replacement

        text = re.sub(slash_command_pattern, replace_command, text)
        text = re.sub(r'glyph<[^>]*>', replace_glyph, text)
        text = re.sub(r'/([A-Z])\.cap', replace_cap, text)

        return text, occurrences_amount, corrections
    
    def _block_ends_with_colon(self, block):
        """Check if block text ends with colon for relevant block types."""
        block_type = block.get("type")
        text = block.get("text", "").rstrip()
        if block_type in {"text", "caption", "section_header", "paragraph"}:
            return text.endswith(":")
        return False

    def _apply_formatting_rules(self, blocks):
        """把过滤后的块序列渲染成带标题层级与分组的 markdown 文本。

        规则引擎（状态机式逐块消费，i 指针非单调 +1 即可一次吞掉整个「组」）:
            - 页眉/章节头 -> #(报告级) 或 ##(节级) 标题：以「是否位于页面前 3 块」
              启发式判断层级 —— 年报首页顶部出现的多为标题，中后部的页眉只是跑题头；
            - paragraph -> ### 小节（其文本即节名）；以冒号结尾且紧邻表格/列表的
              paragraph 转为该组的前导标题，与表格/列表一起渲染；
            - table / list_item 及其前导标题、后随脚注聚成「组」渲染，
              保证表格不孤悬、脚注不与其正文失联；
            - 未知块类型 raise（schema 漂移时 fail-fast，而不是静默漏内容）。

        Args:
            blocks: 页内块列表（已滤除页脚/图片等忽略类型）

        Returns:
            按渲染顺序拼好的文本块列表（块间以换行连接）。
        """
        page_header_in_first_3 = False
        section_header_in_first_3 = False
        # 只看页面前 3 块: 判定本页是否为「标题页」—— 决定后面标题用 H1 还是 H2
        for blk in blocks[:3]:
            if blk["type"] == "page_header":
                page_header_in_first_3 = True
            if blk["type"] == "section_header":
                section_header_in_first_3 = True

        final_blocks = []
        first_section_header_index = 0

        i = 0
        n = len(blocks)

        while i < n:
            block = blocks[i]
            block_type = block.get("type")
            text = block.get("text", "").strip()

            # Handle headers
            if block_type == "page_header":
                prefix = "\n# " if i < 3 else "\n## "
                final_blocks.append(f"{prefix}{text}\n")
                i += 1
                continue

            if block_type == "section_header":
                first_section_header_index += 1
                if (
                    first_section_header_index == 1
                    and i < 3
                    and not page_header_in_first_3
                ):
                    prefix = "\n# "
                else:
                    prefix = "\n## "
                final_blocks.append(f"{prefix}{text}\n")
                i += 1
                continue

            if block_type == "paragraph":
                # 两个分支渲染相同 —— 有效语义: 普通 paragraph 一律作 ### 小节标题；
                # 仅当「冒号结尾且下一块是 table/list_item」时不在此分支消费，
                # 落入下方分组逻辑作为表格/列表的前导标题（此时不输出独立 ###）。
                if self._block_ends_with_colon(block) and i + 1 < n:
                    next_block_type = blocks[i + 1].get("type")
                    if next_block_type not in ("table", "list_item"):
                        final_blocks.append(f"\n### {text}\n")
                        i += 1
                        continue
                else:
                    final_blocks.append(f"\n### {text}\n")
                    i += 1
                    continue

            # Handle table groups: 入口 = 表格块本体，或「冒号结尾的前导行+下一块为表格」
            # （前导行作组标题并入渲染，见下方 header_for_table 逻辑）
            if block_type == "table" or (
                self._block_ends_with_colon(block)
                and i + 1 < n
                and blocks[i + 1].get("type") == "table"
            ):
                group_blocks = []
                header_for_table = None
                if self._block_ends_with_colon(block) and i + 1 < n:
                    header_for_table = block
                    table_block = blocks[i + 1]
                    i += 2
                else:
                    table_block = block
                    i += 1

                if header_for_table:
                    group_blocks.append(header_for_table)
                group_blocks.append(table_block)

                footnote_candidates_start = i
                if i < n:
                    maybe_text_block = blocks[i]
                    if maybe_text_block.get("type") == "text":
                        if (i + 1 < n) and (blocks[i + 1].get("type") == "footnote"):
                            group_blocks.append(maybe_text_block)
                            i += 1

                while i < n and blocks[i].get("type") == "footnote":
                    group_blocks.append(blocks[i])
                    i += 1

                group_text = self._render_table_group(group_blocks)
                final_blocks.append(group_text)
                continue

            # Handle list groups: 与表格同构 —— 列表与其前导标题、尾随脚注打包渲染，
            # 列表项统一加 "- " 前缀输出，保证切块后仍是合法 markdown 列表
            if block_type == "list_item" or (
                self._block_ends_with_colon(block)
                and i + 1 < n
                and blocks[i + 1].get("type") == "list_item"
            ):
                group_blocks = []
                if self._block_ends_with_colon(block) and i + 1 < n:
                    header_for_list = block
                    i += 1
                    group_blocks.append(header_for_list)

                while i < n and blocks[i].get("type") == "list_item":
                    group_blocks.append(blocks[i])
                    i += 1

                if i < n and blocks[i].get("type") == "text":
                    if (i + 1 < n) and (blocks[i + 1].get("type") == "footnote"):
                        group_blocks.append(blocks[i])
                        i += 1

                while i < n and blocks[i].get("type") == "footnote":
                    group_blocks.append(blocks[i])
                    i += 1

                group_text = self._render_list_group(group_blocks)
                final_blocks.append(group_text)
                continue

            # Handle normal blocks
            if block_type in (
                "text",
                "caption",
                "footnote",
                "checkbox_selected",
                "checkbox_unselected",
                "formula",
            ):
                if not text.strip():
                    i += 1
                    continue
                else:
                    final_blocks.append(f"{text}\n")
                    i += 1
                continue

            # 未知类型直接炸: 01 schema 出现新标签说明上游漂移，静默丢弃会悄悄丢内容，
            # 而检索语料缺一块就可能漏一个可答点 —— fail-fast 优于带病续跑
            raise ValueError(f"Unknown block type: {block_type}")

        return final_blocks

    def _render_table_group(self, group_blocks):
        """Render table group with optional header, text and footnotes."""
        chunk = []
        for blk in group_blocks:
            blk_type = blk.get("type")
            blk_text = blk.get("text", "").strip()
            if blk_type in {"text", "caption", "section_header", "paragraph"}:
                chunk.append(f"{blk_text}\n")

            elif blk_type == "table":
                table_id = blk.get("table_id")
                if table_id is None:
                    continue
                table_markdown = self._get_table_by_id(table_id)
                chunk.append(f"{table_markdown}\n")

            elif blk_type == "footnote":
                chunk.append(f"{blk_text}\n")

            elif blk_type == "text":
                chunk.append(f"{blk_text}\n")

            else:
                raise ValueError(f"Unexpected block type in table group: {blk_type}")

        return "\n" + "".join(chunk) + "\n"

    def _render_list_group(self, group_blocks):
        """Render list group with optional header, text and footnotes."""
        chunk = []
        for blk in group_blocks:
            blk_type = blk.get("type")
            blk_text = blk.get("text", "").strip()
            if blk_type in {"text", "caption", "section_header", "paragraph"}:
                chunk.append(f"{blk_text}\n")

            elif blk_type == "list_item":
                chunk.append(f"- {blk_text}\n")

            elif blk_type == "footnote":
                chunk.append(f"{blk_text}\n")

            elif blk_type == "checkbox_selected":
                chunk.append(f"[x] {blk_text}\n")

            elif blk_type == "checkbox_unselected":
                chunk.append(f"[ ] {blk_text}\n")

            else:
                chunk.append(f"{blk_text}\n")

        return "\n" + "".join(chunk) + "\n"

    def _get_table_by_id(self, table_id):
        """Get table representation by ID from report data.
        Returns markdown or serialized text based on configuration."""
        for t in self.report_data.get("tables", []):
            if t.get("table_id") == table_id:
                if self.use_serialized_tables:
                    return self._get_serialized_table_text(t, self.serialized_tables_instead_of_markdown)
                return t.get("markdown", "")
        raise ValueError(f"Table with ID={table_id} not found in report_data!")
    
    def _get_serialized_table_text(self, table, serialized_tables_instead_of_markdown):
        """Convert serialized table format to text string.
        
        Args:
            table: Table object containing serialized data
            
        Returns:
            String containing concatenated information blocks or markdown as fallback
        """
        if not table.get("serialized"):
            return table.get("markdown", "")
            
        info_blocks = table["serialized"].get("information_blocks", [])
        text_blocks = [block["information_block"] for block in info_blocks]
        serialized_text = "\n".join(text_blocks)
        if serialized_tables_instead_of_markdown:
            return serialized_text
        else:
            markdown = table.get("markdown", "")
            combined_text = f"{markdown}\nDescription of the table entities:\n{serialized_text}"
            return combined_text

    def export_to_markdown(self, reports_dir: Path, output_dir: Path):
        """Export processed reports to markdown files.
        
        Args:
            reports_dir: Directory containing JSON report files
            output_dir: Directory where markdown files will be saved
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for report_path in reports_dir.glob("*.json"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            processed_report = self.process_report(report_data)
            
            document_text = ""
            for page in processed_report['pages']:
                document_text += f"\n\n---\n\n# Page {page['page']}\n\n"
                document_text += page['text']
            
            report_name = report_data['metainfo']['sha1_name']
            with open(output_dir / f"{report_name}.md", "w", encoding="utf-8") as f:
                f.write(document_text)
