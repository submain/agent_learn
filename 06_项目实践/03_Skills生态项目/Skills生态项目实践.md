# Skills生态项目实践

## 项目概述

本阶段通过一个完整的Skills生态系统项目，深入实践Skills的开发、注册、发现、使用和市场功能。该项目展示如何构建一个可扩展、可维护的Skills生态系统。

## 项目：Skills生态系统平台

### 项目背景
构建一个Skills生态系统平台，支持Skill的开发者创建、注册、发布和销售Skills，同时支持用户发现、购买和使用Skills。

### 技术栈
- LangChain 1.0 (langchain-core, langchain-openai)
- DeepSeek API
- FastAPI（Web服务）
- SQLite（数据库）
- Skills注册表
- Skills市场

### 项目结构

```
skills_ecosystem/
├── config.py                    # 配置文件
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── base_skill.py           # 基础Skill类
│   ├── skill_registry.py       # Skill注册表
│   └── skill_market.py         # Skill市场
├── skills/                      # Skill实现
│   ├── __init__.py
│   ├── search_skill.py         # 搜索Skill
│   ├── calculation_skill.py    # 计算Skill
│   ├── data_analysis_skill.py  # 数据分析Skill
│   └── automation_skill.py     # 自动化Skill
├── api/                         # API接口
│   ├── __init__.py
│   ├── skill_api.py            # Skill API
│   ├── market_api.py           # 市场API
│   └── agent_api.py            # 智能体API
├── database/                    # 数据库
│   ├── __init__.py
│   ├── models.py               # 数据模型
│   └── repository.py           # 数据仓库
├── agents/                      # 智能体
│   ├── __init__.py
│   ├── skill_agent.py          # Skill智能体
│   └── market_agent.py         # 市场智能体
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── validators.py           # 验证器
│   └── formatters.py           # 格式化器
└── main.py                      # 主程序入口
```

### 核心代码实现

#### 1. 基础Skill类 (core/base_skill.py)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SkillMetadata:
    """Skill元数据"""
    name: str
    description: str
    version: str
    author: str
    category: str
    tags: List[str]
    price: float
    rating: float
    downloads: int
    created_at: str
    updated_at: str

class BaseSkill(ABC):
    """基础Skill类"""
    
    def __init__(self, metadata: SkillMetadata):
        self.metadata = metadata
        self._is_installed = False
        self._is_activated = False
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行Skill"""
        pass
    
    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """验证输入"""
        pass
    
    def install(self) -> bool:
        """安装Skill"""
        if self._is_installed:
            return False
        
        self._is_installed = True
        return True
    
    def uninstall(self) -> bool:
        """卸载Skill"""
        if not self._is_installed:
            return False
        
        self._is_installed = False
        self._is_activated = False
        return True
    
    def activate(self) -> bool:
        """激活Skill"""
        if not self._is_installed:
            return False
        
        self._is_activated = True
        return True
    
    def deactivate(self) -> bool:
        """停用Skill"""
        if not self._is_activated:
            return False
        
        self._is_activated = False
        return True
    
    def is_installed(self) -> bool:
        """检查是否已安装"""
        return self._is_installed
    
    def is_activated(self) -> bool:
        """检查是否已激活"""
        return self._is_activated
    
    def get_info(self) -> Dict[str, Any]:
        """获取Skill信息"""
        return {
            "metadata": self.metadata.__dict__,
            "status": {
                "installed": self._is_installed,
                "activated": self._is_activated
            }
        }
```

#### 2. Skill注册表 (core/skill_registry.py)

```python
from typing import Dict, List, Optional
from core.base_skill import BaseSkill, SkillMetadata
import json
from datetime import datetime

class SkillRegistry:
    """Skill注册表"""
    
    def __init__(self):
        self.skills: Dict[str, BaseSkill] = {}
        self.installed_skills: Dict[str, BaseSkill] = {}
        self.activated_skills: Dict[str, BaseSkill] = {}
    
    def register(self, skill: BaseSkill) -> bool:
        """注册Skill"""
        skill_name = skill.metadata.name
        
        if skill_name in self.skills:
            return False
        
        self.skills[skill_name] = skill
        return True
    
    def unregister(self, skill_name: str) -> bool:
        """注销Skill"""
        if skill_name not in self.skills:
            return False
        
        skill = self.skills[skill_name]
        
        if skill.is_installed():
            skill.uninstall()
        
        del self.skills[skill_name]
        return True
    
    def install(self, skill_name: str) -> bool:
        """安装Skill"""
        if skill_name not in self.skills:
            return False
        
        skill = self.skills[skill_name]
        
        if not skill.install():
            return False
        
        self.installed_skills[skill_name] = skill
        return True
    
    def uninstall(self, skill_name: str) -> bool:
        """卸载Skill"""
        if skill_name not in self.installed_skills:
            return False
        
        skill = self.installed_skills[skill_name]
        
        if not skill.uninstall():
            return False
        
        del self.installed_skills[skill_name]
        
        if skill_name in self.activated_skills:
            del self.activated_skills[skill_name]
        
        return True
    
    def activate(self, skill_name: str) -> bool:
        """激活Skill"""
        if skill_name not in self.installed_skills:
            return False
        
        skill = self.installed_skills[skill_name]
        
        if not skill.activate():
            return False
        
        self.activated_skills[skill_name] = skill
        return True
    
    def deactivate(self, skill_name: str) -> bool:
        """停用Skill"""
        if skill_name not in self.activated_skills:
            return False
        
        skill = self.activated_skills[skill_name]
        
        if not skill.deactivate():
            return False
        
        del self.activated_skills[skill_name]
        return True
    
    def get_skill(self, skill_name: str) -> Optional[BaseSkill]:
        """获取Skill"""
        return self.skills.get(skill_name)
    
    def list_skills(self, category: Optional[str] = None) -> List[Dict]:
        """列出所有Skill"""
        skills_list = []
        
        for skill in self.skills.values():
            if category is None or skill.metadata.category == category:
                skills_list.append(skill.get_info())
        
        return skills_list
    
    def search_skills(self, query: str) -> List[Dict]:
        """搜索Skill"""
        results = []
        query_lower = query.lower()
        
        for skill in self.skills.values():
            metadata = skill.metadata
            
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append(skill.get_info())
        
        return results
    
    def get_activated_skills(self) -> List[BaseSkill]:
        """获取已激活的Skill"""
        return list(self.activated_skills.values())
    
    def execute_skill(self, skill_name: str, **kwargs) -> Any:
        """执行Skill"""
        if skill_name not in self.activated_skills:
            raise ValueError(f"Skill '{skill_name}' 未激活")
        
        skill = self.activated_skills[skill_name]
        
        if not skill.validate_input(**kwargs):
            raise ValueError("输入验证失败")
        
        return skill.execute(**kwargs)
```

#### 3. Skill市场 (core/skill_market.py)

```python
from typing import Dict, List, Optional
from core.base_skill import BaseSkill, SkillMetadata
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MarketSkill:
    """市场Skill"""
    metadata: SkillMetadata
    download_url: str
    documentation_url: str
    reviews: List[Dict]
    total_downloads: int
    
    def get_average_rating(self) -> float:
        """获取平均评分"""
        if not self.reviews:
            return 0.0
        
        total = sum(review["rating"] for review in self.reviews)
        return total / len(self.reviews)

class SkillMarket:
    """Skill市场"""
    
    def __init__(self):
        self.marketplace: Dict[str, MarketSkill] = {}
        self._initialize_marketplace()
    
    def _initialize_marketplace(self):
        """初始化市场"""
        sample_skills = [
            {
                "name": "web_search",
                "description": "强大的网络搜索工具，支持多种搜索引擎",
                "version": "1.0.0",
                "author": "SkillTeam",
                "category": "搜索",
                "tags": ["搜索", "网络", "信息检索"],
                "price": 0.0,
                "rating": 4.8,
                "downloads": 15234,
                "download_url": "https://market.skills.com/web_search.zip",
                "documentation_url": "https://docs.skills.com/web_search"
            },
            {
                "name": "data_analysis",
                "description": "数据分析工具，支持统计分析和可视化",
                "version": "2.1.0",
                "author": "DataExpert",
                "category": "数据分析",
                "tags": ["数据分析", "统计", "可视化"],
                "price": 9.99,
                "rating": 4.6,
                "downloads": 8765,
                "download_url": "https://market.skills.com/data_analysis.zip",
                "documentation_url": "https://docs.skills.com/data_analysis"
            },
            {
                "name": "automation",
                "description": "自动化任务执行工具，支持工作流编排",
                "version": "1.5.0",
                "author": "AutoMaster",
                "category": "自动化",
                "tags": ["自动化", "工作流", "任务执行"],
                "price": 19.99,
                "rating": 4.7,
                "downloads": 5432,
                "download_url": "https://market.skills.com/automation.zip",
                "documentation_url": "https://docs.skills.com/automation"
            }
        ]
        
        for skill_data in sample_skills:
            metadata = SkillMetadata(
                name=skill_data["name"],
                description=skill_data["description"],
                version=skill_data["version"],
                author=skill_data["author"],
                category=skill_data["category"],
                tags=skill_data["tags"],
                price=skill_data["price"],
                rating=skill_data["rating"],
                downloads=skill_data["downloads"],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            market_skill = MarketSkill(
                metadata=metadata,
                download_url=skill_data["download_url"],
                documentation_url=skill_data["documentation_url"],
                reviews=[],
                total_downloads=skill_data["downloads"]
            )
            
            self.marketplace[skill_data["name"]] = market_skill
    
    def list_skills(self, category: Optional[str] = None, 
                   sort_by: str = "rating") -> List[Dict]:
        """列出市场Skill"""
        skills = list(self.marketplace.values())
        
        if category:
            skills = [s for s in skills if s.metadata.category == category]
        
        if sort_by == "rating":
            skills.sort(key=lambda x: x.get_average_rating(), reverse=True)
        elif sort_by == "downloads":
            skills.sort(key=lambda x: x.total_downloads, reverse=True)
        elif sort_by == "price":
            skills.sort(key=lambda x: x.metadata.price)
        
        return [self._market_skill_to_dict(s) for s in skills]
    
    def search_skills(self, query: str) -> List[Dict]:
        """搜索市场Skill"""
        results = []
        query_lower = query.lower()
        
        for market_skill in self.marketplace.values():
            metadata = market_skill.metadata
            
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.description.lower() or
                any(query_lower in tag.lower() for tag in metadata.tags)):
                results.append(self._market_skill_to_dict(market_skill))
        
        return results
    
    def get_skill_details(self, skill_name: str) -> Optional[Dict]:
        """获取Skill详情"""
        if skill_name not in self.marketplace:
            return None
        
        return self._market_skill_to_dict(self.marketplace[skill_name])
    
    def add_review(self, skill_name: str, user: str, rating: int, comment: str):
        """添加评价"""
        if skill_name not in self.marketplace:
            return False
        
        review = {
            "user": user,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        
        self.marketplace[skill_name].reviews.append(review)
        return True
    
    def get_reviews(self, skill_name: str) -> List[Dict]:
        """获取评价"""
        if skill_name not in self.marketplace:
            return []
        
        return self.marketplace[skill_name].reviews
    
    def _market_skill_to_dict(self, market_skill: MarketSkill) -> Dict:
        """转换为字典"""
        return {
            "metadata": market_skill.metadata.__dict__,
            "average_rating": market_skill.get_average_rating(),
            "total_downloads": market_skill.total_downloads,
            "download_url": market_skill.download_url,
            "documentation_url": market_skill.documentation_url,
            "review_count": len(market_skill.reviews)
        }
```

#### 4. 搜索Skill实现 (skills/search_skill.py)

```python
from core.base_skill import BaseSkill, SkillMetadata
from typing import Dict, Any
import requests

class SearchSkill(BaseSkill):
    """搜索Skill"""
    
    def __init__(self):
        metadata = SkillMetadata(
            name="web_search",
            description="强大的网络搜索工具，支持多种搜索引擎",
            version="1.0.0",
            author="SkillTeam",
            category="搜索",
            tags=["搜索", "网络", "信息检索"],
            price=0.0,
            rating=4.8,
            downloads=15234,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )
        super().__init__(metadata)
    
    def validate_input(self, **kwargs) -> bool:
        """验证输入"""
        if "query" not in kwargs:
            return False
        
        if not isinstance(kwargs["query"], str):
            return False
        
        if len(kwargs["query"]) == 0:
            return False
        
        return True
    
    def execute(self, **kwargs) -> Any:
        """执行搜索"""
        query = kwargs["query"]
        engine = kwargs.get("engine", "google")
        num_results = kwargs.get("num_results", 5)
        
        results = self._perform_search(query, engine, num_results)
        
        return {
            "query": query,
            "engine": engine,
            "results": results,
            "count": len(results)
        }
    
    def _perform_search(self, query: str, engine: str, num_results: int) -> list:
        """执行实际搜索"""
        mock_results = [
            {
                "title": f"关于'{query}'的结果1",
                "url": "https://example.com/1",
                "snippet": f"这是关于{query}的搜索结果摘要..."
            },
            {
                "title": f"关于'{query}'的结果2",
                "url": "https://example.com/2",
                "snippet": f"这是关于{query}的另一个搜索结果..."
            },
            {
                "title": f"关于'{query}'的结果3",
                "url": "https://example.com/3",
                "snippet": f"更多关于{query}的信息..."
            }
        ]
        
        return mock_results[:num_results]
```

#### 5. 计算Skill实现 (skills/calculation_skill.py)

```python
from core.base_skill import BaseSkill, SkillMetadata
from typing import Dict, Any
import math

class CalculationSkill(BaseSkill):
    """计算Skill"""
    
    def __init__(self):
        metadata = SkillMetadata(
            name="calculation",
            description="数学计算工具，支持基本运算和高级函数",
            version="1.0.0",
            author="MathExpert",
            category="计算",
            tags=["计算", "数学", "运算"],
            price=0.0,
            rating=4.5,
            downloads=9876,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )
        super().__init__(metadata)
    
    def validate_input(self, **kwargs) -> bool:
        """验证输入"""
        if "operation" not in kwargs:
            return False
        
        valid_operations = ["add", "subtract", "multiply", "divide", "power", "sqrt", "sin", "cos", "tan"]
        
        if kwargs["operation"] not in valid_operations:
            return False
        
        return True
    
    def execute(self, **kwargs) -> Any:
        """执行计算"""
        operation = kwargs["operation"]
        
        if operation == "add":
            return self._add(kwargs.get("a", 0), kwargs.get("b", 0))
        elif operation == "subtract":
            return self._subtract(kwargs.get("a", 0), kwargs.get("b", 0))
        elif operation == "multiply":
            return self._multiply(kwargs.get("a", 0), kwargs.get("b", 0))
        elif operation == "divide":
            return self._divide(kwargs.get("a", 0), kwargs.get("b", 1))
        elif operation == "power":
            return self._power(kwargs.get("a", 0), kwargs.get("b", 0))
        elif operation == "sqrt":
            return self._sqrt(kwargs.get("a", 0))
        elif operation == "sin":
            return self._sin(kwargs.get("a", 0))
        elif operation == "cos":
            return self._cos(kwargs.get("a", 0))
        elif operation == "tan":
            return self._tan(kwargs.get("a", 0))
    
    def _add(self, a: float, b: float) -> float:
        return a + b
    
    def _subtract(self, a: float, b: float) -> float:
        return a - b
    
    def _multiply(self, a: float, b: float) -> float:
        return a * b
    
    def _divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("除数不能为零")
        return a / b
    
    def _power(self, a: float, b: float) -> float:
        return math.pow(a, b)
    
    def _sqrt(self, a: float) -> float:
        if a < 0:
            raise ValueError("负数不能开平方")
        return math.sqrt(a)
    
    def _sin(self, a: float) -> float:
        return math.sin(a)
    
    def _cos(self, a: float) -> float:
        return math.cos(a)
    
    def _tan(self, a: float) -> float:
        return math.tan(a)
```

#### 6. 数据分析Skill实现 (skills/data_analysis_skill.py)

```python
from core.base_skill import BaseSkill, SkillMetadata
from typing import Dict, Any, List
import statistics

class DataAnalysisSkill(BaseSkill):
    """数据分析Skill"""
    
    def __init__(self):
        metadata = SkillMetadata(
            name="data_analysis",
            description="数据分析工具，支持统计分析和可视化",
            version="2.1.0",
            author="DataExpert",
            category="数据分析",
            tags=["数据分析", "统计", "可视化"],
            price=9.99,
            rating=4.6,
            downloads=8765,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )
        super().__init__(metadata)
    
    def validate_input(self, **kwargs) -> bool:
        """验证输入"""
        if "data" not in kwargs:
            return False
        
        if not isinstance(kwargs["data"], list):
            return False
        
        if len(kwargs["data"]) == 0:
            return False
        
        return True
    
    def execute(self, **kwargs) -> Any:
        """执行数据分析"""
        data = kwargs["data"]
        operation = kwargs.get("operation", "summary")
        
        if operation == "summary":
            return self._summary_statistics(data)
        elif operation == "correlation":
            return self._correlation_analysis(data)
        elif operation == "distribution":
            return self._distribution_analysis(data)
        else:
            return self._summary_statistics(data)
    
    def _summary_statistics(self, data: List[float]) -> Dict[str, float]:
        """摘要统计"""
        return {
            "count": len(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "mode": statistics.mode(data) if len(set(data)) < len(data) else None,
            "std_dev": statistics.stdev(data) if len(data) > 1 else 0,
            "variance": statistics.variance(data) if len(data) > 1 else 0,
            "min": min(data),
            "max": max(data),
            "range": max(data) - min(data)
        }
    
    def _correlation_analysis(self, data: List[List[float]]) -> Dict[str, float]:
        """相关性分析"""
        if len(data) < 2:
            return {"error": "需要至少两组数据"}
        
        x = data[0]
        y = data[1]
        
        correlation = statistics.correlation(x, y) if len(x) == len(y) and len(x) > 1 else 0
        
        return {
            "correlation": correlation,
            "interpretation": self._interpret_correlation(correlation)
        }
    
    def _interpret_correlation(self, correlation: float) -> str:
        """解释相关性"""
        if abs(correlation) > 0.8:
            return "强相关"
        elif abs(correlation) > 0.5:
            return "中等相关"
        elif abs(correlation) > 0.3:
            return "弱相关"
        else:
            return "几乎不相关"
    
    def _distribution_analysis(self, data: List[float]) -> Dict[str, Any]:
        """分布分析"""
        sorted_data = sorted(data)
        n = len(data)
        
        q1 = sorted_data[n // 4]
        q2 = sorted_data[n // 2]
        q3 = sorted_data[3 * n // 4]
        
        iqr = q3 - q1
        
        outliers = [x for x in data if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]
        
        return {
            "quartiles": {
                "q1": q1,
                "q2": q2,
                "q3": q3
            },
            "iqr": iqr,
            "outliers": outliers,
            "outlier_count": len(outliers)
        }
```

#### 7. 自动化Skill实现 (skills/automation_skill.py)

```python
from core.base_skill import BaseSkill, SkillMetadata
from typing import Dict, Any, List
import time

class AutomationSkill(BaseSkill):
    """自动化Skill"""
    
    def __init__(self):
        metadata = SkillMetadata(
            name="automation",
            description="自动化任务执行工具，支持工作流编排",
            version="1.5.0",
            author="AutoMaster",
            category="自动化",
            tags=["自动化", "工作流", "任务执行"],
            price=19.99,
            rating=4.7,
            downloads=5432,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )
        super().__init__(metadata)
    
    def validate_input(self, **kwargs) -> bool:
        """验证输入"""
        if "tasks" not in kwargs:
            return False
        
        if not isinstance(kwargs["tasks"], list):
            return False
        
        if len(kwargs["tasks"]) == 0:
            return False
        
        return True
    
    def execute(self, **kwargs) -> Any:
        """执行自动化任务"""
        tasks = kwargs["tasks"]
        mode = kwargs.get("mode", "sequential")
        
        if mode == "sequential":
            return self._execute_sequential(tasks)
        elif mode == "parallel":
            return self._execute_parallel(tasks)
        else:
            return self._execute_sequential(tasks)
    
    def _execute_sequential(self, tasks: List[Dict]) -> Dict[str, Any]:
        """顺序执行任务"""
        results = []
        start_time = time.time()
        
        for i, task in enumerate(tasks):
            task_result = self._execute_single_task(task)
            results.append({
                "task_id": i + 1,
                "task": task,
                "result": task_result,
                "status": "success" if task_result["success"] else "failed"
            })
        
        end_time = time.time()
        
        return {
            "mode": "sequential",
            "total_tasks": len(tasks),
            "successful_tasks": sum(1 for r in results if r["status"] == "success"),
            "failed_tasks": sum(1 for r in results if r["status"] == "failed"),
            "execution_time": end_time - start_time,
            "results": results
        }
    
    def _execute_parallel(self, tasks: List[Dict]) -> Dict[str, Any]:
        """并行执行任务"""
        import concurrent.futures
        
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(self._execute_single_task, task): (i + 1, task)
                for i, task in enumerate(tasks)
            }
            
            for future in concurrent.futures.as_completed(future_to_task):
                task_id, task = future_to_task[future]
                try:
                    task_result = future.result()
                    results.append({
                        "task_id": task_id,
                        "task": task,
                        "result": task_result,
                        "status": "success" if task_result["success"] else "failed"
                    })
                except Exception as e:
                    results.append({
                        "task_id": task_id,
                        "task": task,
                        "result": {"success": False, "error": str(e)},
                        "status": "failed"
                    })
        
        end_time = time.time()
        
        return {
            "mode": "parallel",
            "total_tasks": len(tasks),
            "successful_tasks": sum(1 for r in results if r["status"] == "success"),
            "failed_tasks": sum(1 for r in results if r["status"] == "failed"),
            "execution_time": end_time - start_time,
            "results": results
        }
    
    def _execute_single_task(self, task: Dict) -> Dict[str, Any]:
        """执行单个任务"""
        task_type = task.get("type", "default")
        
        try:
            if task_type == "sleep":
                time.sleep(task.get("duration", 1))
                return {"success": True, "message": f"Sleep {task.get('duration', 1)}s"}
            elif task_type == "print":
                print(task.get("message", "Hello"))
                return {"success": True, "message": task.get("message", "Hello")}
            elif task_type == "calculate":
                a = task.get("a", 0)
                b = task.get("b", 0)
                operation = task.get("operation", "add")
                
                if operation == "add":
                    result = a + b
                elif operation == "multiply":
                    result = a * b
                else:
                    result = 0
                
                return {"success": True, "result": result}
            else:
                return {"success": True, "message": "Task completed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
```

#### 8. Skill智能体 (agents/skill_agent.py)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from core.skill_registry import SkillRegistry
from typing import Dict, Any
from config import Config

class SkillAgent:
    """Skill智能体"""
    
    def __init__(self, skill_registry: SkillRegistry):
        self.skill_registry = skill_registry
        
        self.llm = ChatOpenAI(
            model=Config.DEEPSEEK_MODEL,
            api_key=Config.DEEPSEEK_API_KEY,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            base_url="https://api.deepseek.com"
        )
        
        self.system_prompt = """你是一个智能助手，可以使用各种Skills来完成任务。

可用的Skills：
{skills_info}

当用户请求时，请：
1. 理解用户需求
2. 选择合适的Skill
3. 调用Skill执行任务
4. 返回结果给用户

如果需要使用Skill，请使用以下格式：
USE_SKILL: skill_name
PARAMETERS: {param1: value1, param2: value2}
"""
    
    def update_skills_info(self):
        """更新Skills信息"""
        activated_skills = self.skill_registry.get_activated_skills()
        
        skills_info = "\n".join([
            f"- {skill.metadata.name}: {skill.metadata.description}"
            for skill in activated_skills
        ])
        
        self.system_prompt = self.system_prompt.format(skills_info=skills_info)
    
    def process_request(self, user_input: str) -> str:
        """处理用户请求"""
        self.update_skills_info()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", user_input)
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        
        response_text = response.content
        
        if "USE_SKILL:" in response_text:
            return self._execute_skill(response_text)
        
        return response_text
    
    def _execute_skill(self, response_text: str) -> str:
        """执行Skill"""
        try:
            skill_name = response_text.split("USE_SKILL:")[1].split("\n")[0].strip()
            
            parameters_text = response_text.split("PARAMETERS:")[1].split("\n")[0].strip()
            
            import ast
            parameters = ast.literal_eval(parameters_text)
            
            result = self.skill_registry.execute_skill(skill_name, **parameters)
            
            return f"Skill执行结果：{result}"
        except Exception as e:
            return f"Skill执行失败：{str(e)}"
```

#### 9. 主程序 (main.py)

```python
from core.skill_registry import SkillRegistry
from core.skill_market import SkillMarket
from skills.search_skill import SearchSkill
from skills.calculation_skill import CalculationSkill
from skills.data_analysis_skill import DataAnalysisSkill
from skills.automation_skill import AutomationSkill
from agents.skill_agent import SkillAgent

def main():
    print("=" * 60)
    print("Skills生态系统平台")
    print("=" * 60)
    
    skill_registry = SkillRegistry()
    skill_market = SkillMarket()
    
    search_skill = SearchSkill()
    calculation_skill = CalculationSkill()
    data_analysis_skill = DataAnalysisSkill()
    automation_skill = AutomationSkill()
    
    skill_registry.register(search_skill)
    skill_registry.register(calculation_skill)
    skill_registry.register(data_analysis_skill)
    skill_registry.register(automation_skill)
    
    print("\n可用的Skills：")
    for skill_info in skill_registry.list_skills():
        print(f"- {skill_info['metadata']['name']}: {skill_info['metadata']['description']}")
    
    print("\n市场Skills：")
    for skill_info in skill_market.list_skills():
        print(f"- {skill_info['metadata']['name']}: {skill_info['metadata']['description']} (评分: {skill_info['average_rating']})")
    
    print("\n" + "=" * 60)
    print("输入 'install <skill_name>' 安装Skill")
    print("输入 'activate <skill_name>' 激活Skill")
    print("输入 'use <skill_name> <parameters>' 使用Skill")
    print("输入 'market' 查看市场")
    print("输入 'quit' 退出程序")
    print("=" * 60)
    
    skill_agent = SkillAgent(skill_registry)
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("感谢使用，再见！")
            break
        
        if user_input.startswith("install "):
            skill_name = user_input[8:].strip()
            if skill_registry.install(skill_name):
                print(f"Skill '{skill_name}' 安装成功")
            else:
                print(f"Skill '{skill_name}' 安装失败")
        
        elif user_input.startswith("activate "):
            skill_name = user_input[9:].strip()
            if skill_registry.activate(skill_name):
                print(f"Skill '{skill_name}' 激活成功")
            else:
                print(f"Skill '{skill_name}' 激活失败")
        
        elif user_input.startswith("use "):
            parts = user_input[4:].split()
            if len(parts) >= 1:
                skill_name = parts[0]
                try:
                    import ast
                    parameters = ast.literal_eval(" ".join(parts[1:])) if len(parts) > 1 else {}
                    result = skill_registry.execute_skill(skill_name, **parameters)
                    print(f"结果：{result}")
                except Exception as e:
                    print(f"执行失败：{e}")
        
        elif user_input == "market":
            print("\n市场Skills：")
            for skill_info in skill_market.list_skills():
                print(f"- {skill_info['metadata']['name']}: {skill_info['metadata']['description']}")
                print(f"  评分: {skill_info['average_rating']}, 下载量: {skill_info['total_downloads']}")
        
        elif user_input.startswith("search "):
            query = user_input[7:].strip()
            results = skill_market.search_skills(query)
            print(f"\n搜索结果（'{query}'）：")
            for skill_info in results:
                print(f"- {skill_info['metadata']['name']}: {skill_info['metadata']['description']}")
        
        else:
            response = skill_agent.process_request(user_input)
            print(f"\n助手: {response}")

if __name__ == "__main__":
    main()
```

### 项目特点

1. **模块化设计**：清晰的模块划分和职责分离
2. **可扩展性**：易于添加新的Skill
3. **市场机制**：完整的Skill市场功能
4. **智能集成**：智能体自动选择和使用Skill
5. **灵活配置**：支持Skill的安装、激活、停用

### 运行说明

1. 安装依赖：
```bash
pip install langchain langchain-core langchain-openai python-dotenv fastapi uvicorn
```

2. 配置环境变量：
```bash
DEEPSEEK_API_KEY=your_api_key
```

3. 运行程序：
```bash
python main.py
```

## 学习目标

通过完成这个Skills生态系统项目，你将掌握：

1. **Skill架构设计**：理解Skill的核心架构和设计模式
2. **Skill开发**：学会开发和实现各种类型的Skill
3. **注册表管理**：掌握Skill的注册、安装、激活机制
4. **市场机制**：理解Skill市场的运作模式
5. **智能集成**：学会将Skill集成到智能体中
6. **生态构建**：了解如何构建完整的Skills生态系统

## 扩展练习

1. 添加Skill版本管理和更新机制
2. 实现Skill的权限和访问控制
3. 添加Skill的性能监控和优化
4. 实现Skill的依赖管理
5. 构建Skill的开发者工具链

## 参考资源

- [LangChain Tools 文档](https://python.langchain.com/docs/modules/tools/)
- [Skill开发最佳实践](https://python.langchain.com/docs/modules/tools/custom_tools/)
- [Agent开发指南](https://python.langchain.com/docs/modules/agents/)
- [插件系统设计模式](https://en.wikipedia.org/wiki/Plug-in_(computing))
