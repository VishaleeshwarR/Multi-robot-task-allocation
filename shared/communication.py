"""
Communication protocols between robots and supervisor
"""
import json
import struct
from typing import Dict, Any, List, Tuple
from shared.constants import MessageType

class Message:
    """Base message class"""
    def __init__(self, msg_type: MessageType, data: Dict[str, Any]):
        self.type = msg_type
        self.data = data
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        msg_dict = {
            'type': self.type.value,
            'data': self.data
        }
        return json.dumps(msg_dict).encode('utf-8')
    
    @staticmethod
    def from_bytes(data: bytes) -> 'Message':
        """Deserialize message from bytes"""
        msg_dict = json.loads(data.decode('utf-8'))
        msg_type = MessageType(msg_dict['type'])
        return Message(msg_type, msg_dict['data'])

class SupervisorBroadcast:
    """Supervisor broadcasts global state"""
    @staticmethod
    def create_state_message(q: List[float], x: List[float], 
                            time: float) -> Message:
        return Message(MessageType.STATE_UPDATE, {
            'q': q,
            'x': x,
            'time': time
        })

class RobotReport:
    """Robot reports status to supervisor"""
    @staticmethod
    def create_task_change_message(robot_id: int, old_task: int, 
                                   new_task: int) -> Message:
        return Message(MessageType.TASK_CHANGE, {
            'robot_id': robot_id,
            'old_task': old_task,
            'new_task': new_task
        })
    
    @staticmethod
    def create_status_message(robot_id: int, current_task: int,
                            basket_count: int, position: Tuple[float, float],
                            is_active: bool) -> Message:
        return Message(MessageType.ROBOT_STATUS, {
            'robot_id': robot_id,
            'current_task': current_task,
            'basket_count': basket_count,
            'position': position,
            'is_active': is_active
        })