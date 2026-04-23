import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from src.preprocessing.background_removal import BackgroundRemover
from src.preprocessing.edge_extraction import StructuralExtractor

def test_background_remover_init():
    with patch("rembg.new_session") as mock_new_session:
        remover = BackgroundRemover(model_name="u2net")
        mock_new_session.assert_called_once_with("u2net")
        assert remover.session == mock_new_session.return_value

def test_background_remover_remove_background():
    with patch("rembg.new_session"), patch("rembg.remove") as mock_remove:
        remover = BackgroundRemover()
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        
        # mock returns an RGBA image
        mock_rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        mock_remove.return_value = mock_rgba
        
        product_rgba, mask = remover.remove_background(img)
        
        assert product_rgba.mode == "RGBA"
        assert mask.mode == "L"
        assert mask.size == (100, 100)

def test_structural_extractor_canny():
    extractor = StructuralExtractor(method="canny")
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    
    # Add a white square to trigger edges
    import cv2
    arr = np.array(img)
    arr[25:75, 25:75] = [255, 255, 255]
    img_with_square = Image.fromarray(arr)
    
    edge_map = extractor.extract(img_with_square)
    assert edge_map.mode == "L"
    assert edge_map.size == (100, 100)
    
    # Check that there are edges
    edge_arr = np.array(edge_map)
    assert np.any(edge_arr > 0)

def test_structural_extractor_for_controlnet():
    extractor = StructuralExtractor(method="canny")
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    edge_map_rgb = extractor.extract_for_controlnet(img)
    assert edge_map_rgb.mode == "RGB"
    assert edge_map_rgb.size == (100, 100)
