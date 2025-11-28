#!/usr/bin/env python3
"""
红绿灯分类结果评估脚本
计算准确率、召回率、F1分数，并生成混淆矩阵
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 类别映射
FOLDER_TO_CLASS = {
    'red': 'RED',
    'yellow': 'YELLOW',
    'green': 'GREEN'
}

CLASS_NAMES = ['RED', 'YELLOW', 'GREEN', 'OFF']

def load_ground_truth(dataset_path):
    """从数据集文件夹结构加载真实标签"""
    gt_labels = {}
    
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path):
            continue
            
        # 获取该文件夹下所有图片
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            pattern = os.path.join(folder_path, ext)
            for img_path in glob.glob(pattern):
                # 提取文件名（不含扩展名）
                filename = os.path.splitext(os.path.basename(img_path))[0]
                gt_labels[filename] = class_name
    
    return gt_labels

def load_predictions(output_path):
    """加载预测结果"""
    predictions = {}
    
    for txt_file in glob.glob(os.path.join(output_path, '*.txt')):
        filename = os.path.splitext(os.path.basename(txt_file))[0]
        with open(txt_file, 'r') as f:
            pred_class = f.read().strip()
            predictions[filename] = pred_class
    
    return predictions

def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热图
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置刻度
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix - Traffic Light Classification')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def save_markdown_report(output_path, dataset_path, pred_path, y_true, y_pred, 
                        accuracy, precision, recall, f1, 
                        macro_precision, macro_recall, macro_f1,
                        weighted_precision, weighted_recall, weighted_f1,
                        cm):
    """保存Markdown格式的评估报告"""
    report_path = os.path.join(output_path, 'evaluation_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Traffic Light Classification Evaluation Report\n\n")
        
        # 基本信息
        f.write("## Dataset Information\n\n")
        f.write(f"- **Dataset Path**: `{dataset_path}`\n")
        f.write(f"- **Predictions Path**: `{pred_path}`\n")
        f.write(f"- **Total Samples**: {len(y_true)}\n")
        f.write(f"- **Evaluation Date**: {os.popen('date').read().strip()}\n\n")
        
        # 总体准确率
        f.write("## Overall Performance\n\n")
        f.write(f"**Accuracy**: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        # 每类指标表格
        f.write("## Per-Class Metrics\n\n")
        f.write("| Class | Precision | Recall | F1-Score |\n")
        f.write("|-------|-----------|--------|----------|\n")
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(f"| {class_name} | {precision[i]:.4f} | {recall[i]:.4f} | {f1[i]:.4f} |\n")
        
        # 平均指标
        f.write("\n## Average Metrics\n\n")
        f.write("| Average Type | Precision | Recall | F1-Score |\n")
        f.write("|--------------|-----------|--------|----------|\n")
        f.write(f"| Macro Avg | {macro_precision:.4f} | {macro_recall:.4f} | {macro_f1:.4f} |\n")
        f.write(f"| Weighted Avg | {weighted_precision:.4f} | {weighted_recall:.4f} | {weighted_f1:.4f} |\n")
        
        # 混淆矩阵
        f.write("\n## Confusion Matrix\n\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        
        f.write("### Confusion Matrix (Raw Values)\n\n")
        f.write("|  | " + " | ".join(CLASS_NAMES) + " |\n")
        f.write("|" + "|".join(["-" * 10] * (len(CLASS_NAMES) + 1)) + "|\n")
        for i, class_name in enumerate(CLASS_NAMES):
            row = f"| **{class_name}** |"
            for j in range(len(CLASS_NAMES)):
                row += f" {cm[i, j]} |"
            f.write(row + "\n")
        
        # 详细分类报告
        f.write("\n## Detailed Classification Report\n\n")
        f.write("```\n")
        f.write(classification_report(y_true, y_pred, labels=CLASS_NAMES, zero_division=0))
        f.write("```\n")
        
        # 样本分布
        f.write("\n## Sample Distribution\n\n")
        from collections import Counter
        gt_dist = Counter(y_true)
        pred_dist = Counter(y_pred)
        
        f.write("### Ground Truth Distribution\n\n")
        f.write("| Class | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        for class_name in CLASS_NAMES:
            count = gt_dist.get(class_name, 0)
            pct = (count / len(y_true) * 100) if len(y_true) > 0 else 0
            f.write(f"| {class_name} | {count} | {pct:.2f}% |\n")
        
        f.write("\n### Prediction Distribution\n\n")
        f.write("| Class | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        for class_name in CLASS_NAMES:
            count = pred_dist.get(class_name, 0)
            pct = (count / len(y_pred) * 100) if len(y_pred) > 0 else 0
            f.write(f"| {class_name} | {count} | {pct:.2f}% |\n")
    
    print(f"Markdown report saved to: {report_path}")

def main():
    # 配置路径
    dataset_path = '/home/liming/datasets/cropped_traffic_light'
    output_path = '/home/liming/workspace/mobileclip_cpp_trt/outputs/traffic_light'
    
    print("="*60)
    print("Traffic Light Classification Evaluation")
    print("="*60)
    
    # 加载真实标签
    print("\nLoading ground truth labels...")
    gt_labels = load_ground_truth(dataset_path)
    print(f"Loaded {len(gt_labels)} ground truth labels")
    
    # 加载预测结果
    print("\nLoading predictions...")
    predictions = load_predictions(output_path)
    print(f"Loaded {len(predictions)} predictions")
    
    # 匹配标签和预测
    y_true = []
    y_pred = []
    
    for filename in gt_labels.keys():
        if filename in predictions:
            y_true.append(gt_labels[filename])
            y_pred.append(predictions[filename])
    
    print(f"\nMatched {len(y_true)} samples for evaluation")
    
    if len(y_true) == 0:
        print("Error: No matched samples found!")
        return
    
    # 计算指标
    print("\n" + "="*60)
    print("Classification Metrics")
    print("="*60)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 每个类别的精确率、召回率、F1分数
    precision = precision_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=CLASS_NAMES, average=None, zero_division=0)
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    
    # 宏平均和加权平均
    print("\n" + "-" * 60)
    macro_precision = precision_score(y_true, y_pred, labels=CLASS_NAMES, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=CLASS_NAMES, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=CLASS_NAMES, average='macro', zero_division=0)
    
    weighted_precision = precision_score(y_true, y_pred, labels=CLASS_NAMES, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, labels=CLASS_NAMES, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=CLASS_NAMES, average='weighted', zero_division=0)
    
    print(f"{'Macro Avg':<10} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
    print(f"{'Weighted Avg':<10} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
    print("-" * 60)
    
    # 详细分类报告
    print("\n" + "="*60)
    print("Detailed Classification Report")
    print("="*60)
    print(classification_report(y_true, y_pred, labels=CLASS_NAMES, zero_division=0))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    print("\n" + "="*60)
    print("Confusion Matrix")
    print("="*60)
    print(cm)
    
    # 绘制并保存混淆矩阵
    cm_save_path = os.path.join(output_path, 'confusion_matrix.png')
    plot_confusion_matrix(cm, CLASS_NAMES, cm_save_path)
    
    # 保存Markdown报告
    save_markdown_report(output_path, dataset_path, output_path, y_true, y_pred,
                        accuracy, precision, recall, f1,
                        macro_precision, macro_recall, macro_f1,
                        weighted_precision, weighted_recall, weighted_f1,
                        cm)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_path}")
    print("="*60)

if __name__ == '__main__':
    main()
