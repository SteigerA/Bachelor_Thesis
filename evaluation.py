import os
import json
import statistics
from bert_score import score
from rouge import Rouge

def normalize_text(text):
    """Normalize text for comparison"""
    if text is None:
        return None
    text = str(text).strip().lower()
    if "not specified" in text or "unspecified" in text:
        return "not specified"
    return text

def main():
    results = {
        'overall': {'TP': 0, 'FP': 0, 'FN': 0},  # Track global TP/FP/FN
        'performance': {
            'processing_times': [],
            'embedding_counts': []
        },
        'field_scores': {
            'issuer': {'TP': 0, 'FP': 0, 'FN': 0},
            'category': {'TP': 0, 'FP': 0, 'FN': 0},
            'submission_deadline': {'TP': 0, 'FP': 0, 'FN': 0},
            'end_date': {'TP': 0, 'FP': 0, 'FN': 0}
        },
        'description_scores': {
            'bert_f1': [],
            'rouge_1_f1': [],
            'rouge_2_f1': [],
            'rouge_l_f1': []
        }
    }

    field_mappings = [
        ('issuer', 'Institution', 'issuer'),
        ('category', 'service_category', 'category'),
        ('submission_deadline', 'submission_deadline', 'deadline_for_offer_submission'),
        ('end_date', 'end_date', 'contract_end_date')
    ]

    # Initialize ROUGE scorer
    rouge_scorer = Rouge()

    for entry in os.scandir("./extracted"):
        tender_name = os.path.basename(entry.path)
        try:
            with open(f"./extracted/{tender_name}") as f, open(f"./ground/{tender_name}") as g:
                extracted = json.load(f)
                ground_truth = json.load(g)

                # Track performance metrics
                if '_metrics' in extracted:
                    results['performance']['processing_times'].append(
                        extracted['_metrics']['processing_time_sec']
                    )
                    results['performance']['embedding_counts'].append(
                        extracted['_metrics']['embedding_count']
                    )

                # Evaluate each field
                for field_name, extracted_key, ground_truth_key in field_mappings:
                    e_val = normalize_text(extracted.get(extracted_key))
                    g_val = normalize_text(ground_truth.get(ground_truth_key))

                    # True Positive
                    if e_val and g_val and e_val == g_val:
                        results['overall']['TP'] += 1
                        results['field_scores'][field_name]['TP'] += 1
                    # False Positive (wrong extraction)
                    elif e_val and g_val and e_val != g_val:
                        results['overall']['FP'] += 1
                        results['field_scores'][field_name]['FP'] += 1
                    # False Negative (missed extraction)
                    elif not e_val and g_val:
                        results['overall']['FN'] += 1
                        results['field_scores'][field_name]['FN'] += 1

                # BERT Score and ROUGE Score for description
                extracted_desc = extracted.get('service_summary', '')
                ground_truth_desc = ground_truth.get('service_description', '')
                if extracted_desc and ground_truth_desc:
                    # BERT Score
                    _, _, f1 = score([extracted_desc], [ground_truth_desc], lang='en', verbose=False, rescale_with_baseline=True)
                    results['description_scores']['bert_f1'].append(f1.mean().item())
                    
                    # ROUGE Score
                    try:
                        rouge_scores = rouge_scorer.get_scores(extracted_desc, ground_truth_desc)[0]
                        results['description_scores']['rouge_1_f1'].append(rouge_scores['rouge-1']['f'])
                        results['description_scores']['rouge_2_f1'].append(rouge_scores['rouge-2']['f'])
                        results['description_scores']['rouge_l_f1'].append(rouge_scores['rouge-l']['f'])
                    except Exception as e:
                        print(f"ROUGE calculation failed for {tender_name}: {e}")
                        # Add zeros to maintain consistent list lengths
                        results['description_scores']['rouge_1_f1'].append(0)
                        results['description_scores']['rouge_2_f1'].append(0)
                        results['description_scores']['rouge_l_f1'].append(0)

        except Exception as e:
            print(f"Skipping {tender_name}: {e}")
            continue

    # Calculate overall F1
    tp, fp, fn = results['overall']['TP'], results['overall']['FP'], results['overall']['FN']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    overall_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate description scores
    bert_samples = len(results['description_scores']['bert_f1'])
    rouge_samples = len(results['description_scores']['rouge_1_f1'])
    
    avg_bert_f1 = statistics.mean(results['description_scores']['bert_f1']) * 100 if bert_samples > 0 else 0
    avg_rouge_1_f1 = statistics.mean(results['description_scores']['rouge_1_f1']) * 100 if rouge_samples > 0 else 0
    avg_rouge_2_f1 = statistics.mean(results['description_scores']['rouge_2_f1']) * 100 if rouge_samples > 0 else 0
    avg_rouge_l_f1 = statistics.mean(results['description_scores']['rouge_l_f1']) * 100 if rouge_samples > 0 else 0

    # Generate output
    output = {
        'overall_f1': f"{overall_f1:.1%}",
        'overall_precision': f"{precision:.1%}",
        'overall_recall': f"{recall:.1%}",
        'description_quality': {
            'average_bert_f1': f"{avg_bert_f1:.1f}%",
            'average_rouge_1_f1': f"{avg_rouge_1_f1:.1f}%",
            'average_rouge_2_f1': f"{avg_rouge_2_f1:.1f}%",
            'average_rouge_l_f1': f"{avg_rouge_l_f1:.1f}%",
            'samples': bert_samples
        },
        'performance': {
            'average_processing_time_sec': f"{statistics.mean(results['performance']['processing_times']):.1f}" if results['performance']['processing_times'] else "N/A",
            'average_embedding_count': int(statistics.mean(results['performance']['embedding_counts'])) if results['performance']['embedding_counts'] else "N/A"
        },
        'field_performance': {
            field: {
                'f1': f"{2 * (s['TP']/(s['TP']+s['FP'])) * (s['TP']/(s['TP']+s['FN'])) / ((s['TP']/(s['TP']+s['FP'])) + (s['TP']/(s['TP']+s['FN']))) if (s['TP']+s['FP']) > 0 and (s['TP']+s['FN']) > 0 else 0:.1%}",
                'precision': f"{s['TP']/(s['TP']+s['FP']) if (s['TP']+s['FP']) > 0 else 0:.1%}",
                'recall': f"{s['TP']/(s['TP']+s['FN']) if (s['TP']+s['FN']) > 0 else 0:.1%}",
                'samples': s['TP'] + s['FP'] + s['FN']
            }
            for field, s in results['field_scores'].items()
        }
    }

    with open('evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Evaluation complete. Overall F1: {output['overall_f1']}")
    print(f"BERT F1: {output['description_quality']['average_bert_f1']}")
    print(f"ROUGE-1 F1: {output['description_quality']['average_rouge_1_f1']}")
    print(f"ROUGE-2 F1: {output['description_quality']['average_rouge_2_f1']}")
    print(f"ROUGE-L F1: {output['description_quality']['average_rouge_l_f1']}")

if __name__ == "__main__":
    main()