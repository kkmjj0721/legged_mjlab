from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Optional, Any


class FSMState(ABC):
    """FSM状态机类
    """
    def __init__(self, name: str):
        self._state_name = name

    @abstractmethod
    def enter(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def exit(self) -> None:
        pass

    def check_change(self) -> str:
        """默认返回当前状态名，表示不切换状态"""
        return self._state_name

    @property
    def state_name(self) -> str:
        return self._state_name

class FSMMode(Enum):
    """FSM运行模式
    """
    NORMAL = auto()
    CHANGE = auto()

class FSM:
    """有限状态机核心类
    """
    def __init__(self):
        self.states: Dict[str, FSMState] = {}
        self.current_state: Optional[FSMState] = None
        self.next_state: Optional[FSMState] = None
        self.previous_state: Optional[FSMState] = None
        self.mode: FSMMode = FSMMode.NORMAL

    def add_state(self, state: FSMState) -> None:
        self.states[state.state_name] = state

    def set_initial_state(self, name: str) -> None:
        self.current_state = self.states[name]
        self.current_state.enter()
        self.next_state = self.current_state
        print(f"[INFO] [FSM] Set initial state: {name}")

    def request_state_change(self, state_name: str) -> None:
        if state_name not in self.states:
            print(f"[ERROR] [FSM] State '{state_name}' not found!")
            return

        if self.current_state and self.current_state.state_name != state_name:
            self.next_state = self.states[state_name]
            self.mode = FSMMode.CHANGE
            print(f"\n[INFO] [FSM] Request switch from {self.current_state.state_name} to {self.next_state.state_name}")

    def run(self) -> None:
        if not self.current_state:
            return

        if self.mode == FSMMode.NORMAL:
            self.current_state.run()
            next_state_name = self.current_state.check_change()
            if next_state_name != self.current_state.state_name:
                self.mode = FSMMode.CHANGE
                self.next_state = self.states[next_state_name]
                print(f"\n[NOTE] [FSM] Switch from {self.current_state.state_name} to {self.next_state.state_name}")

        elif self.mode == FSMMode.CHANGE:
            self.current_state.exit()
            self.previous_state = self.current_state
            self.current_state = self.next_state
            expected_state = self.next_state  # Save the state we're entering
            self.current_state.enter()

            if self.mode == FSMMode.CHANGE and self.next_state != expected_state:
                return

            self.mode = FSMMode.NORMAL
            self.current_state.run()

class FSMFactory(ABC):
    """FSM工厂基类
    """
    @abstractmethod
    def create_state(self, context: Any, state_name: str) -> Optional[FSMState]:
        pass

    @abstractmethod
    def get_type(self) -> str:
        pass

    @abstractmethod
    def get_supported_states(self) -> List[str]:
        pass

    @abstractmethod
    def get_initial_state(self) -> str:
        pass

class FSMManager:
    """FSM管理器（单例模式）
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FSMManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.factories: Dict[str, FSMFactory] = {}
        return cls._instance

    def register_factory(self, factory: FSMFactory) -> None:
        if factory:
            factory_type = factory.get_type()
            self.factories[factory_type] = factory
            print(f"[INFO] [FSMManager] Registered type: {factory_type}")

    def create_fsm(self, fsm_type: str, context: Any) -> Optional[FSM]:
        factory = self.factories.get(fsm_type)
        if not factory:
            print(f"[ERROR] [FSMManager] Error: Unsupported type: {fsm_type}")
            return None
        
        state_names = factory.get_supported_states()
        if not state_names:
            print(f"[ERROR] [FSMManager] Error: No states registered for type: {fsm_type}")
            return None
            
        fsm = FSM()
        for state_name in state_names:
            state = factory.create_state(context, state_name)
            if state:
                fsm.add_state(state)
                
        fsm.set_initial_state(factory.get_initial_state())
        print(f"[INFO] [FSMManager] FSM created for type: {fsm_type}")
        return fsm

    def is_type_supported(self, fsm_type: str) -> bool:
        return fsm_type in self.factories

    def get_supported_types(self) -> List[str]:
        return list(self.factories.keys())

def register_fsm_factory(initial_state_name: str):
    """
    用作替代 C++ 中 REGISTER_FSM_FACTORY 宏的装饰器。
    它会自动实例化被装饰的工厂类，并注册到单例 FSMManager 中。
    """
    def decorator(cls):
        # 实例化工厂对象并注册
        factory_instance = cls(initial_state_name)
        FSMManager().register_factory(factory_instance)
        return cls
    return decorator