import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from src.pipeline.enhancement_pipeline import ClearShotPipeline, EnhancementResult

@pytest.fixture
def mock_pipeline():
    with patch("src.pipeline.enhancement_pipeline.ClearShotPipeline._load_config") as mock_load:
        mock_load.return_value = {
            "pipeline": {"num_inference_steps": 2},
            "prompt": {"template": "test", "negative": "test"},
            "background": {"mode": "white"},
            "super_resolution": {"enabled": True}
        }
        pipeline = ClearShotPipeline(device="cpu")
        
        # Mock submodules to prevent lazy loading heavy dependencies
        pipeline._bg_remover = MagicMock()
        img = Image.new("RGBA", (512, 512), (255, 0, 0, 255))
        mask = Image.new("L", (512, 512), 255)
        pipeline._bg_remover.remove_background.return_value = (img, mask)
        pipeline._bg_remover.apply_studio_background.return_value = img.convert("RGB")
        
        pipeline._edge_extractor = MagicMock()
        pipeline._edge_extractor.extract_for_controlnet.return_value = Image.new("RGB", (512, 512), (0, 0, 0))
        
        pipeline._diffusion_enhancer = MagicMock()
        pipeline._diffusion_enhancer.enhance.return_value = Image.new("RGB", (512, 512), (0, 255, 0))
        pipeline._diffusion_enhancer.lora_loaded = False
        
        pipeline._super_resolver = MagicMock()
        pipeline._super_resolver.upscale.return_value = Image.new("RGB", (1024, 1024), (0, 255, 0))
        
        return pipeline

def test_pipeline_enhance(mock_pipeline):
    input_img = Image.new("RGB", (256, 256), (128, 128, 128))
    result = mock_pipeline.enhance(input_img)
    
    assert isinstance(result, EnhancementResult)
    assert result.original.size == (256, 256)
    assert result.final.size == (1024, 1024)
    
    assert mock_pipeline._bg_remover.remove_background.called
    assert mock_pipeline._edge_extractor.extract_for_controlnet.called
    assert mock_pipeline._diffusion_enhancer.enhance.called
    assert mock_pipeline._super_resolver.upscale.called
    assert mock_pipeline._bg_remover.apply_studio_background.called
