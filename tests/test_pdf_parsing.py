"""_parse_csv_metadata 的单元测试。

被测对象是 PDFParser 上的纯静态方法（subset.csv 路径 -> {sha1: {company_name}} 查找表），
无需实例化 PDFParser，因此不触发 docling converter 构建与模型加载；但模块顶层
import docling，测试必须跑在装有 docling 的 venv 环境（项目自带 venv）。

运行（项目根目录）:
    .\\venv\\Scripts\\python.exe -m pytest tests\\test_pdf_parsing.py -v
"""
from pathlib import Path

import pytest

from src.pdf_parsing import PDFParser


def _parse(csv_text: str, tmp_path: Path) -> dict:
    """在 pytest 临时目录里造一份 subset.csv 并调用被测函数。"""
    csv_file = tmp_path / "subset.csv"
    csv_file.write_text(csv_text, encoding="utf-8")
    return PDFParser._parse_csv_metadata(csv_file)


@pytest.mark.parametrize(
    "csv_text, expected",
    [
        pytest.param(
            "sha1,company_name\nabc123,ACME Co\n",
            {"abc123": {"company_name": "ACME Co"}},
            id="new-column-basic",
        ),
        pytest.param(
            # 公司名带引号（Excel 手工导出场景）；引号内逗号必须保留
            'sha1,company_name\nabc123,"ACME, Inc."\n',
            {"abc123": {"company_name": "ACME, Inc."}},
            id="strip-outside-quotes-keep-inner-comma",
        ),
        pytest.param(
            # 旧版赛题 CSV 列名为 name，走 row.get 回退链
            "sha1,name\nabc123,OldName Co\n",
            {"abc123": {"company_name": "OldName Co"}},
            id="old-column-name-fallback",
        ),
        pytest.param(
            "sha1,company_name\nabc123,某某集团有限公司\n",
            {"abc123": {"company_name": "某某集团有限公司"}},
            id="utf8-chinese-company-name",
        ),
        pytest.param(
            # DictReader 按表头取列，列顺序与期望无关
            'company_name,sha1\n"ACME, Inc.",abc123\n',
            {"abc123": {"company_name": "ACME, Inc."}},
            id="column-order-irrelevant",
        ),
        pytest.param(
            # 两列都缺失的行 -> company_name 空串（元数据缺失的降级语义）
            "sha1,company_name\nabc123,\n",
            {"abc123": {"company_name": ""}},
            id="missing-name-columns-degrade-to-empty",
        ),
        pytest.param(
            # Windows 导出的 CRLF 行尾
            "sha1,company_name\r\nabc123,ACME Co\r\n",
            {"abc123": {"company_name": "ACME Co"}},
            id="crlf-line-endings",
        ),
        pytest.param(
            # 空文件 -> 空查找表
            "",
            {},
            id="empty-file",
        ),
    ],
)
def test_parse_csv_metadata(csv_text: str, expected: dict, tmp_path: Path):
    assert _parse(csv_text, tmp_path) == expected


def test_duplicate_sha1_last_row_wins(tmp_path: Path):
    """dict 赋值语义：同一 sha1 出现多行时后者覆盖前者（实际 CSV 不应出现，仅固化行为）。"""
    csv_text = "sha1,company_name\nabc123,Old Co\nabc123,New Co\n"
    assert _parse(csv_text, tmp_path) == {"abc123": {"company_name": "New Co"}}


def test_missing_sha1_column_raises_keyerror(tmp_path: Path):
    """实现假定 CSV 必有 sha1 列（下游按文件名=sha1 匹配），缺列直接 KeyError，不静默。"""
    csv_text = "company_name\nACME Co\n"
    with pytest.raises(KeyError):
        _parse(csv_text, tmp_path)
