import torch
import json
from models import get_model, count_parameters
from train import train_model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    models_config = [
        ('baseline', '1_Baseline_CNN'),
        ('pure_lnn', '2_Pure_LNN'),
        ('lnn_cbam', '3_LNN_CBAM'),
        ('hybrid', '4_Hybrid_CNN_LNN'),
        ('hybrid_cbam', '5_Hybrid_CNN_LNN_CBAM'),
    ]
    
    results = {}
    
    for model_type, save_name in models_config:
        print(f"\n{'#'*60}")
        print(f"# Training: {save_name}")
        print(f"{'#'*60}\n")
        
        model = get_model(model_type)
        params = count_parameters(model)
        
        history = train_model(
            model=model,
            model_name=save_name,
            device=device,
            epochs=10,
            batch_size=128,
            lr=0.001
        )
        
        # Save results
        results[save_name] = {
            'model_type': model_type,
            'parameters': params,
            'final_train_acc': history['train_acc'][-1],
            'best_test_acc': history['best_test_acc'],  
            'best_epoch': history['best_epoch'], 
            'final_test_acc': history['test_acc'][-1], 
            'history': history
        }
        
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Completed: {save_name}")
        print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
        print(f"Parameters: {params:,}")
        print(f"{'='*60}\n")
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Parameters':>15} {'Train Acc':>12} {'Test Acc':>12}")
    print("-"*80)
    
    baseline_acc = results['1_Baseline_CNN']['best_test_acc']

    for name, data in results.items():
        clean_name = name[2:]
        params = data['parameters'] / 1e6
        train_acc = data['final_train_acc']
        test_acc = data['best_test_acc'] 
        best_epoch = data['best_epoch'] 
        diff = test_acc - baseline_acc
    
        print(f"{clean_name:<30} {params:>13.2f}M {train_acc:>11.2f}% {test_acc:>11.2f}% (Epoch {best_epoch}) ({diff:+.1f}%)")
    
    print("="*80)
    print("\nResults saved to 'results.json'")
    print("Model weights saved as '<model_name>.pth'")


if __name__ == '__main__':
    main()