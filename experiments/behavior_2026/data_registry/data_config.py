from starVLA.dataloader.gr00t_lerobot.data_config import DataConfig

# 2026 BEHAVIOR 챌린지 표준인 R1Pro 로봇 및 4채널(RGB-D) 모달리티 설정
ROBOT_TYPE_CONFIG_MAP = {
    "r1pro": DataConfig(
        image_channels=4,  # RGB(3) + Depth(1) 4채널 입력 대응
        state_dim=32,      # R1Pro 고유수용성 감각 상태 차원
        action_dim=23,     # R1Pro 액션 제어 차원
    )
}

DATASET_NAMED_MIXTURES = {
    "behavior_2026_train": [
        ("behavior-1k/2026-challenge-demos", 1.0, "r1pro"),
    ]
}
