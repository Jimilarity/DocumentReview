import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .filtering import _select_rules
from .normalization import load_rule_aliases, normalize_rules
from .rule_type import decode_rule_type
from utils import read_json


@dataclass
class RuleSet:
    rules: List[Dict[str, Any]]


RuleFilter = Callable[
    [List[Dict[str, Any]], Dict[str, bool]],
    List[Dict[str, Any]],
]


class RuleSetBuilder:
    def __init__(self, candidate_rules: Dict[str, Any]) -> None:
        self.candidate_rules = candidate_rules
        self.rule_aliases = load_rule_aliases()
        self.rules: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}

    @classmethod
    def from_json(cls, path: str | Path) -> "RuleSetBuilder":
        return cls(read_json(path))

    def _rules_for_type(
        self,
        rule_type: int,
    ) -> List[Dict[str, Any]]:
        self.config = decode_rule_type(rule_type)
        selected_rules: List[Dict[str, Any]] = []
        if self.config["use_legality"]:
            legality_rules = self.candidate_rules["合法性标准"]
            selected_rules.extend(legality_rules.get("通用", []))
            selected_rules.extend(_select_rules(legality_rules, self.config))

        if self.config["use_standardization"]:
            selected_rules.extend(
                _select_rules(
                    self.candidate_rules["规范性标准"],
                    self.config,
                )
            )

        if self.config["use_additional_review"]:
            selected_rules.extend(
                self.candidate_rules.get("附加项", {}).get("通用", [])
            )

        return normalize_rules(selected_rules, self.rule_aliases)

    def for_rule_type(self, rule_type: int) -> "RuleSetBuilder":
        """只按业务规则类型选取候选规则，不掺入执行器信息。"""

        self.rules = copy.deepcopy(self._rules_for_type(rule_type))
        return self

    def for_executor(
        self,
        rule_type: int,
        *,
        rule_filter: RuleFilter,
        document_presence: Dict[str, bool],
    ) -> "RuleSetBuilder":
        """使用叶子执行器提供的策略物化一份独立规则集。"""

        selected_rules = self._rules_for_type(rule_type)
        self.rules = rule_filter(selected_rules, document_presence)
        return self

    def build(self) -> RuleSet:
        return RuleSet(rules=copy.deepcopy(self.rules))
