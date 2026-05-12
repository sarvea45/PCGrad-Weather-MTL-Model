import os
from model import build_mtl_model

def main():
    os.makedirs('results', exist_ok=True)
    model = build_mtl_model()
    
    with open('results/model_architecture.txt', 'w') as f:
        f.write("=== Shared Backbone Summary ===\n")
        model.backbone.summary(print_fn=lambda x: f.write(x + '\n'))
        
        f.write("\n=== Full Model Summary ===\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    print("Model summary generated at results/model_architecture.txt")

if __name__ == "__main__":
    main()
