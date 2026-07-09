from langchain.agents import create_agent
from typing import List
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Annotated, Dict, List, Literal
from model_config import build_text_model, build_vision_model
from utils import load_yaml

class caseMeta(BaseModel):
    law_enforcement_unit : str = Field(alias="执法单位")
    case_name : str = Field(alias="案卷名称")
    case_number : str = Field(alias="案号")
    cause_of_action : str = Field(alias="案由")

class dirInfo(BaseModel):
    is_dir : bool 
    dir_info : List[Dict]

class sectionInfo(BaseModel):
    is_belong : Dict[str, Literal[-1, 1, 2]]
    
class documentMeta(BaseModel):
    id : int = Field(alias="顺序号")
    document_name : str = Field(alias="文书名称")
    page_number : List = Field(alias="页码")
    
class documentMetaList(BaseModel):
    documentsInfo : List[documentMeta] = Field(alias="文书信息")
    
class ReviewComment(BaseModel):
    section_id: List[int] = Field(default_factory=list)
    content: str = Field(description="具体问题说明")


class ReviewResult(BaseModel):
    comment: List[ReviewComment] = Field(
        default_factory=list,
        description="审查问题列表。若无问题，必须为空列表 []"
    )
    score: float
    confidence: Literal["high", "medium", "low"]



class Documentreview():
    """Documentreview crew"""
    agent_config_path = Path(__file__).parent / "config" / "agents.yaml"
    task_config_path = Path(__file__).parent / "config" / "tasks.yaml"
    agents_config = load_yaml(agent_config_path)
    tasks_config = load_yaml(task_config_path)
    def __init__(self, review_tools = []):
        self.vlm = build_vision_model()
        self.llm = build_text_model()
        self.dir_agent = self.dir_identifier()
        self.section_agent = self.section_identifier()
        self.meta_agent = self.meta_data_extractor()
        self.review_agent = self.document_reviewer(tools=review_tools)    
        self.write_agent = self.result_writer() 
    def build_agent_prompt(self, agent_name):
        config = self.agents_config[agent_name]
        return f"""
    角色：
    {config['role']}
    目标：
    {config['goal']}
    背景：
    {config['backstory']}
    """
    
    def build_task_prompt(self, task_name, **kwargs):
        config = self.tasks_config[task_name]
        description = config["description"].format(**kwargs)
        return f"""
    任务描述：
    {description}
    输出格式：
    {config["expected_output"]}
    """ 
        
    def meta_data_extractor(self) :
        tools = []
        system_prompt = self.build_agent_prompt('meta_data_extractor')
        return create_agent(model=self.vlm, tools=tools, system_prompt=system_prompt)
    
    def dir_identifier(self):
        tools = []
        system_prompt = self.build_agent_prompt('dir_identifier')
        return create_agent(model=self.vlm, tools=tools, system_prompt=system_prompt)
    
    def section_identifier(self):
        tools = []
        system_prompt = self.build_agent_prompt('section_identifier')
        return create_agent(model=self.vlm, tools=tools, system_prompt=system_prompt)
        
    
    def document_reviewer(self, tools = []):
        system_prompt = self.build_agent_prompt("document_reviewer")
        if len(tools) > 1:
            system_prompt = system_prompt + f"""涉及中国法律法规检索时：
            1. 必须使用 get_law_list 工具，禁止凭记忆罗列法规名
            2. 解读结果时优先采用:在法律文书所记录的违法行为发生的时间段生效的文书，可能文书所引用的法律在现在已经失效，但在当时生效
            3. 引用法规时必须带上 DocumentNO（发文字号）+ Url（pkulaw.com 原文链接）以便溯源
            4. 涉及多部法规效力冲突时，根据 EffectivenessDic（效力级别）判断优先级：法律 > 行政法规 > 部门规章 > 司法解释性文件
            5. 找不到匹配法规时坦诚告知"未检索到相关法规"，建议用户换关键词或改用「检索法律法规-语义」服务用自然语言案情查
            6.要拿单条法规的完整法条原文，请配合订阅「检索法律法规-语义」服务的 get_article 工具（按法律名+条号查）
            """
        model_with_tools = self.llm.bind_tools(tools) 
        return create_agent(model=model_with_tools, system_prompt=system_prompt)
    
    def result_writer(self):
        return self.llm.bind(
        response_format={"type": "json_object"},
        extra_body={
            "enable_thinking": False
        }
    )
        
        
    




    