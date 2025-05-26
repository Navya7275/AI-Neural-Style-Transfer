import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import copy

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loader
def image_loader(image_path, imsize=(512, 512)):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Style transfer loss functions
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Normalization layer
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std

# Model builder
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                style_img, content_img,
                                content_layers=['conv_4'],
                                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):

    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j + 1]

    return model, style_losses, content_losses

# Style transfer runner
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    run = [0]

    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight * style_score + content_weight * content_score
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# Final function for Gradio
def stylize(content_path, style_path, progress=gr.Progress()):
    progress(0, desc="Initializing models...")
    
    content_img = image_loader(content_path)
    style_img = image_loader(style_path)
    input_img = content_img.clone()

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    progress(0.2, desc="Processing style transfer...")
    
    # Run style transfer with fewer steps for demo purposes
    output = run_style_transfer(cnn, normalization_mean, normalization_std,
                                content_img, style_img, input_img, num_steps=200)

    progress(0.9, desc="Finalizing image...")
    output_image = output.squeeze().cpu().clone().clamp(0, 1)
    return transforms.ToPILImage()(output_image)

# Improved CSS with lilac theme
css = """
/* Main Theme Colors - Lilac Palette */
:root {
  --lilac-light: #E8E0FF;
  --lilac-medium: #C5B3FF;
  --lilac-dark: #9277FF;
  --lilac-accent: #7C4DFF;
  --lilac-deep: #5E35B1;
  --text-dark: #483D8B;
  --text-light: #F8F7FF;
  --accent-pink: #FF80AB;
  --accent-pink-hover: #FF4081;
}

/* Global Styles */
body {
  font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

.gradio-container {
  background: linear-gradient(135deg, var(--lilac-light) 0%, #F5F0FF 100%);
  color: var(--text-dark);
  max-width: 1200px;
  margin: 0 auto;
}

/* Header Styling - Fixed for clarity */
#app-title {
  color: var(--lilac-deep);
  font-size: 48px;
  font-weight: 800;
  text-align: center;
  margin: 20px auto;
  padding: 10px;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app-subtitle {
  text-align: center;
  margin-bottom: 30px;
  font-size: 18px;
  color: var(--text-dark);
  font-weight: 400;
}

/* Section Styling */
.app-section {
  border-radius: 20px;
  box-shadow: 0 8px 32px rgba(139, 127, 219, 0.15);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  margin-bottom: 2rem;
  overflow: hidden;
  background: white;
}

.app-section:hover {
  transform: translateY(-5px);
  box-shadow: 0 12px 48px rgba(139, 127, 219, 0.25);
}

/* About Section */
#about-section {
  background: linear-gradient(135deg, #FFFFFF 0%, var(--lilac-light) 100%);
  padding: 2rem;
  text-align: center;
}

#about-section h2 {
  color: var(--lilac-deep);
  font-size: 2rem;
  margin-bottom: 1rem;
  position: relative;
  display: inline-block;
}

#about-section h2:after {
  content: "";
  position: absolute;
  bottom: -10px;
  left: 50%;
  transform: translateX(-50%);
  width: 60px;
  height: 4px;
  background: var(--lilac-accent);
  border-radius: 2px;
}

#about-section p {
  color: var(--text-dark);
  font-size: 1.1rem;
  line-height: 1.6;
  max-width: 800px;
  margin: 1.5rem auto 0;
}

/* Features Section */
#features-section {
  background: white;
  padding: 2rem;
  text-align: center;
}

#features-section h2 {
  color: var(--lilac-deep);
  font-size: 2rem;
  margin-bottom: 1.5rem;
  position: relative;
  display: inline-block;
}

#features-section h2:after {
  content: "";
  position: absolute;
  bottom: -10px;
  left: 50%;
  transform: translateX(-50%);
  width: 60px;
  height: 4px;
  background: var(--lilac-accent);
  border-radius: 2px;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
}

.feature-item {
  background: var(--lilac-light);
  padding: 1.5rem;
  border-radius: 12px;
  transition: all 0.3s ease;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.feature-item:hover {
  background: var(--lilac-medium);
  transform: translateY(-5px) scale(1.03);
  box-shadow: 0 10px 25px rgba(139, 127, 219, 0.2);
}

.feature-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  color: var(--lilac-accent);
}

.feature-title {
  font-weight: 600;
  color: var(--lilac-deep);
  margin-bottom: 0.5rem;
}

.feature-text {
  color: var(--text-dark);
  font-size: 0.95rem;
}

/* Upload Interface */
#interface-section {
  background: white;
  padding: 2rem;
  border-radius: 20px;
}

#interface-section h2 {
  color: var(--lilac-deep);
  font-size: 2rem;
  text-align: center;
  margin-bottom: 1.5rem;
  position: relative;
  display: inline-block;
  left: 50%;
  transform: translateX(-50%);
}

#interface-section h2:after {
  content: "";
  position: absolute;
  bottom: -10px;
  left: 50%;
  transform: translateX(-50%);
  width: 60px;
  height: 4px;
  background: var(--lilac-accent);
  border-radius: 2px;
}

/* Image Upload Styling */
.image-upload-container label {
  color: var(--lilac-deep) !important;
  font-weight: 600 !important;
  font-size: 1.1rem !important;
  margin-bottom: 0.75rem !important;
}

.upload-box {
  border: 2px dashed var(--lilac-medium) !important;
  border-radius: 12px !important;
  background-color: var(--lilac-light) !important;
  transition: all 0.3s ease !important;
  padding: 2rem !important;
}

.upload-box:hover {
  border-color: var(--lilac-accent) !important;
  background-color: #F0EBFF !important;
  transform: scale(1.02) !important;
}

/* Button Styling */
#generate-btn {
  background: linear-gradient(45deg, var(--lilac-accent), var(--lilac-deep)) !important;
  color: white !important;
  font-weight: 600 !important;
  font-size: 1.2rem !important;
  padding: 0.75rem 2rem !important;
  border-radius: 12px !important;
  border: none !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 15px rgba(124, 77, 255, 0.3) !important;
  margin: 1.5rem auto !important;
  display: block !important;
  width: fit-content !important;
}

#generate-btn:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 8px 25px rgba(124, 77, 255, 0.4) !important;
  background: linear-gradient(45deg, var(--lilac-deep), var(--lilac-accent)) !important;
}

/* Output Image Container */
.output-container {
  margin-top: 2rem;
  padding: 1.5rem;
  border-radius: 15px;
  background: var(--lilac-light);
  box-shadow: 0 5px 15px rgba(139, 127, 219, 0.1);
  transition: all 0.3s ease;
}

.output-container:hover {
  box-shadow: 0 8px 30px rgba(139, 127, 219, 0.2);
}

.output-container label {
  color: var(--lilac-deep) !important;
  font-weight: 600 !important;
  font-size: 1.1rem !important;
}

/* Footer Styling */
#footer {
  text-align: center;
  padding: 1.5rem;
  color: var(--text-dark);
  font-size: 0.9rem;
  opacity: 0.8;
  margin-top: 2rem;
}

/* Responsiveness */
@media (max-width: 768px) {
  #app-title {
    font-size: 36px;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
  
  #about-section, #features-section, #interface-section {
    padding: 1.5rem;
  }
}
"""

# Define HTML Content Sections - Fixed title
header_html = """
<h1 id="app-title">🎨 Neural Style Transfer</h1>
<div id="app-subtitle">Transform your photos into masterpieces with AI-powered style transfer</div>
"""

about_html = """
<div id="about-section" class="app-section">
  <h2>✨ About This App</h2>
  <p>
    This cutting-edge web application uses <strong>Neural Style Transfer</strong> technology 
    powered by PyTorch and a pre-trained VGG19 model. The algorithm extracts and applies 
    artistic styles from one image to the content of another, creating unique and stunning 
    AI-generated artwork that blends both worlds together.
  </p>
</div>
"""

features_html = """
<div id="features-section" class="app-section">
  <h2>🌟 Features</h2>
  <div class="features-grid">
    <div class="feature-item">
      <div class="feature-icon">🖌️</div>
      <div class="feature-title">Real-time Style Transfer</div>
      <div class="feature-text">Transform your photos with the aesthetic of famous artworks</div>
    </div>
    <div class="feature-item">
      <div class="feature-icon">📸</div>
      <div class="feature-title">Custom Uploads</div>
      <div class="feature-text">Use your own content and style images</div>
    </div>
    <div class="feature-item">
      <div class="feature-icon">⚡</div>
      <div class="feature-title">GPU Acceleration</div>
      <div class="feature-text">Faster processing with CUDA when available</div>
    </div>
    <div class="feature-item">
      <div class="feature-icon">✨</div>
      <div class="feature-title">Beautiful Interface</div>
      <div class="feature-text">Elegant lilac-themed design with modern effects</div>
    </div>
  </div>
</div>
"""

interface_html = """
<div id="interface-section" class="app-section">
  <h2>🎭 Create Your Masterpiece</h2>
</div>
"""

footer_html = """
<div id="footer">
  Created with ❤️ and PyTorch | Neural Style Transfer App
</div>
"""

if __name__ == "__main__":
    with gr.Blocks(css=css) as demo:
        # Header section with fixed title
        gr.HTML(header_html)
        
        # About section (first)
        gr.HTML(about_html)
        
        # Features section (second)
        gr.HTML(features_html)
        
        # Interface section header (third)
        gr.HTML(interface_html)
        
        # Main application interface
        with gr.Row(equal_height=True):
            with gr.Column():
                content_img = gr.Image(
                    type="filepath", 
                    label="👉 Upload Content Image", 
                    elem_id="content-upload",
                    elem_classes="image-upload-container upload-box"
                )
            
            with gr.Column():
                style_img = gr.Image(
                    type="filepath", 
                    label="👉 Upload Style Image", 
                    elem_id="style-upload",
                    elem_classes="image-upload-container upload-box"
                )
                
        # Output section
        output_img = gr.Image(
            label="✨ Your Stylized Masterpiece", 
            elem_classes="output-container"
        )
        
        # Action button
        run_btn = gr.Button(
            "✨ Generate Stylized Image ✨", 
            elem_id="generate-btn"
        )
        run_btn.click(
            fn=stylize, 
            inputs=[content_img, style_img], 
            outputs=output_img
        )
        
        # Footer-
        gr.HTML(footer_html)

    # Launch the app
    demo.launch()
