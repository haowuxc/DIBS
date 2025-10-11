from transformers import CLIPTokenizer, CLIPModel
import torch
import os 

class CLIPTextFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device=None, ckpt_path="/ailab/user/wuhao2/model/ckpts"):
        """
        Initializes the CLIPTextFeatureExtractor.

        Args:
            model_name (str): Name of the CLIP model from Hugging Face's model hub.
            device (str, optional): Device to use for computation ("cuda", "cpu"). Defaults to auto-detection.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the CLIP model and tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(os.path.join(ckpt_path, model_name))
        self.model = CLIPModel.from_pretrained(os.path.join(ckpt_path, model_name)).to(self.device)
    
    def extract_features(self, texts):
        """
        Extracts text features using the CLIP model.

        Args:
            texts (list of str): List of input text strings.

        Returns:
            torch.Tensor: The extracted text features as a tensor.
        """
        # Tokenize the input texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Extract features using the model
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        # Normalize the features to unit length
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

# Example usage
if __name__ == "__main__":
    # Instantiate the extractor
    extractor = CLIPTextFeatureExtractor()

    # Input text examples
    texts = [
        "A futuristic cityscape",
        "A colorful abstract painting",
        "A detailed map of a fantasy world"
    ]
    
    # Extract features
    features = extractor.extract_features(texts)
    
    # Print the extracted features
    print("Extracted text features:")
    print(features.shape)
