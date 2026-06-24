from face_recognition_module_with_Robot import launch_simulation, get_user_status
from conversation_module import run_conversation

pepper = launch_simulation()
status, name = get_user_status(pepper)   # existing face recognition

# NEW: conversation + BN recommendation
result = run_conversation(
    status=status,
    name=name,
    pepper=pepper,
    known_history=None,   # pass stored preference if you have it
)