"""원 저자의 두 신경망 구조와 매개변수 이름을 호환되도록 구현한다."""

import torch
from torch import nn
from torchvision.models import resnet50


class VerificationNetwork(nn.Module):
    """저자 검증 모델: ResNet-50 → 128 → sigmoid → 차이 → FC(1)."""

    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(in_features=2048, out_features=128)
        self.fc_end = nn.Linear(128, 1)

    def encode(self, inputs):
        return torch.sigmoid(self.model(inputs))

    def forward(self, left, right):
        return self.fc_end(torch.abs(self.encode(left) - self.encode(right)))


class ExtractorRes50(nn.Module):
    """분류·전역 풀링 계층을 제거한 저자 검색용 ResNet-50."""

    def __init__(self):
        super().__init__()
        self.extractor_net = nn.Sequential(*list(resnet50(weights=None).children())[:-2])

    def forward(self, inputs):
        return self.extractor_net(inputs)


class AdaptiveConcatPool2d(nn.Module):
    """fastai 2.0.15와 동일한 [MaxPool, AvgPool] 연결 순서."""

    def __init__(self, size=5):
        super().__init__()
        self.mp = nn.AdaptiveMaxPool2d(size)
        self.ap = nn.AdaptiveAvgPool2d(size)

    def forward(self, inputs):
        return torch.cat([self.mp(inputs), self.ap(inputs)], dim=1)


class RetrievalHead(nn.Module):
    """5×5 연결 풀링과 128차원 임베딩을 사용하는 저자의 헤드."""

    def __init__(self):
        super().__init__()
        self.pooling = AdaptiveConcatPool2d(5)
        self.bottle_neck_conv = nn.Conv2d(2048, 100, kernel_size=1, stride=1, bias=False)
        self.first_fc = nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(5000), nn.Dropout(0.5),
            nn.Linear(5000, 2048), nn.ReLU(True),
        )
        self.second_fc = nn.Sequential(
            nn.BatchNorm1d(2048), nn.Dropout(0.25), nn.Linear(2048, 128),
        )

    def forward(self, inputs):
        inputs = self.pooling(inputs)
        batch, channels, height, width = inputs.shape
        inputs = inputs.view(batch, channels // 2, height * 2, width)
        return self.second_fc(self.first_fc(self.bottle_neck_conv(inputs)))


def create_retrieval_network():
    """저자가 nn.Sequential(extractor, head)로 저장한 키 구조를 보존한다."""
    return nn.Sequential(ExtractorRes50(), RetrievalHead())


def model_checkpoint_paths(official_root,verification_checkpoint=None,retrieval_checkpoint=None):
    from pathlib import Path
    root=Path(official_root).parent/'published_models'
    if bool(verification_checkpoint)!=bool(retrieval_checkpoint):raise ValueError('검증·검색 체크포인트를 모두 명시해야 합니다.')
    return {'verification':Path(verification_checkpoint) if verification_checkpoint else root/'verification_approach_final_model.pth',
            'retrieval':Path(retrieval_checkpoint) if retrieval_checkpoint else root/'retrieval_approach_final_model.pth'}


def load_published_models(official_root, device,verification_checkpoint=None,retrieval_checkpoint=None):
    """출판된 모델 가중치를 로드하며 구조가 다르면 즉시 실패한다."""
    official_root = __import__("pathlib").Path(official_root)
    paths=model_checkpoint_paths(official_root,verification_checkpoint,retrieval_checkpoint)
    verification_path=paths['verification'];retrieval_path=paths['retrieval']
    verification = VerificationNetwork()
    retrieval = create_retrieval_network()
    for model, checkpoint in ((verification, verification_path), (retrieval, retrieval_path)):
        if not checkpoint.is_file() or checkpoint.stat().st_size < 100_000:
            raise FileNotFoundError(f"공식 모델이 없습니다. Git LFS 다운로드를 확인하세요: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state,dict) and 'model_state' in state:state=state['model_state']
        elif isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
    return verification, retrieval
