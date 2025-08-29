from __future__ import annotations
import json, re, math
import torch
import pandas as pd
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm

# Load model from local file
MODEL_PATH = "./models/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()

# Qwen chat models prefer left padding for batched generation
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

class ReviewPolicyClassifier:
    def __init__(self):
        self.policy_examples = {
            "No Advertisement": [
                {
                    "review": "Best pizza! Visit www.pizzapromo.com for discounts!",
                    "violation": True
                },
                {
                    "review": "Amazing food and great service. Highly recommend this place!",
                    "violation": False
                }
            ],
            "No Irrelevant Content": [
                {
                    "review": "I love my new phone, but this place is too noisy.",
                    "violation": True
                },
                {
                    "review": "The atmosphere was perfect for a quiet dinner. Food was delicious.",
                    "violation": False
                }
            ],
            "No Rant Without Visit": [
                {
                    "review": "Never been here, but I heard it's terrible.",
                    "violation": True
                },
                {
                    "review": "Visited last week and was disappointed with the slow service.",
                    "violation": False
                }
            ]
        }
    
    def create_few_shot_prompt(self, review_text: str) -> str:
        """Create a single review prompt (for backward compatibility)"""
        return self.create_batch_prompt([review_text])
    
    def create_batch_prompt(self, review_texts: List[str]) -> str:
        """Create a batch prompt for multiple reviews"""
        
        prompt = """<|im_start|>system
You are a review policy classifier. Analyze reviews and determine if they violate these policies:

1. **No Advertisement**: Contains promotional content, links, or advertisements
2. **No Irrelevant Content**: Contains content unrelated to the business/location
3. **No Rant Without Visit**: Contains complaints without indication of actual visit

Examples:
"Best pizza! Visit www.pizzapromo.com for discounts!" → Advertisement: true, Irrelevant: false, Rant: false
"Amazing food and great service. Highly recommend!" → Advertisement: false, Irrelevant: false, Rant: false
"I love my new phone, but this place is too noisy." → Advertisement: false, Irrelevant: true, Rant: false
"Never been here, but I heard it's terrible." → Advertisement: false, Irrelevant: false, Rant: true

For each review, respond with ONLY three booleans: Advertisement, Irrelevant, Rant (true/false).
<|im_end|>

<|im_start|>user
Analyze these reviews:

"""
        
        for i, review in enumerate(review_texts, 1):
            prompt += f"{i}. \"{review}\"\n"
        
        prompt += """\nRespond with only the three booleans for each review, one per line:
Review 1: Advertisement, Irrelevant, Rant
Review 2: Advertisement, Irrelevant, Rant
...<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def generate_with_qwen(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate response using Qwen model (single review compatibility)"""
        return self.generate_with_qwen_batch(prompt, max_new_tokens)
    
    def generate_with_qwen_batch(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate response using Qwen model with reduced token limit"""
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        return response.strip()
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract results from response (for backward compatibility)"""
        # For single review, parse the first line of boolean results
        bool_values = self._extract_booleans_from_line(response.split('\n')[0])
        
        if bool_values:
            return {
                "violations": [
                    {
                        "policy": "No Advertisement",
                        "violated": bool_values[0],
                        "confidence": 0.9,
                        "reason": "Advertisement detected" if bool_values[0] else "No advertisement"
                    },
                    {
                        "policy": "No Irrelevant Content", 
                        "violated": bool_values[1],
                        "confidence": 0.9,
                        "reason": "Irrelevant content detected" if bool_values[1] else "Content relevant"
                    },
                    {
                        "policy": "No Rant Without Visit",
                        "violated": bool_values[2],
                        "confidence": 0.9,
                        "reason": "Rant without visit detected" if bool_values[2] else "No rant without visit"
                    }
                ],
                "overall_compliant": not any(bool_values),
                "summary": f"Violations: {sum(bool_values)}"
            }
        else:
            return self._create_fallback_response(response)
    
    def _create_fallback_response(self, response: str) -> Dict:
        """Create fallback response when JSON parsing fails"""
        return {
            "violations": [
                {
                    "policy": "Error",
                    "violated": False,
                    "confidence": 0.0,
                    "reason": f"Parsing failed. Raw response: {response[:200]}..."
                }
            ],
            "overall_compliant": True,
            "summary": "Error in processing - could not parse model response"
        }
    
    def classify_review(self, review_text: str, max_retries: int = 2) -> Dict:
        """Classify a single review (for backward compatibility)"""
        
        prompt = self.create_few_shot_prompt(review_text)
        
        for attempt in range(max_retries):
            try:
                response = self.generate_with_qwen(prompt)
                parsed_result = self.extract_json_from_response(response)
                return parsed_result
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self._create_error_response(f"Processing error: {e}")
                time.sleep(1)
        
        return self._create_error_response("Max retries exceeded")
    
    def _create_error_response(self, error_msg: str) -> Dict:
        """Create error response in expected format"""
        return {
            "violations": [
                {
                    "policy": "Error",
                    "violated": False,
                    "confidence": 0.0,
                    "reason": error_msg
                }
            ],
            "overall_compliant": True,
            "summary": f"Error in processing: {error_msg}"
        }
    
    def parse_batch_response(self, response: str, num_reviews: int) -> List[Dict]:
        """Parse batch response into individual results"""
        
        results = []
        lines = response.strip().split('\n')
        
        for i in range(num_reviews):
            try:
                # Look for patterns like "Review X:" or just the boolean values
                line_found = False
                for line in lines:
                    if f"Review {i+1}:" in line or f"{i+1}." in line:
                        bool_values = self._extract_booleans_from_line(line)
                        if bool_values:
                            results.append({
                                "advertisement_violation": bool_values[0],
                                "irrelevant_violation": bool_values[1],
                                "rant_violation": bool_values[2],
                                "overall_compliant": not any(bool_values)
                            })
                            line_found = True
                            break
                
                if not line_found:
                    if i < len(lines):
                        bool_values = self._extract_booleans_from_line(lines[i])
                        if bool_values:
                            results.append({
                                "advertisement_violation": bool_values[0],
                                "irrelevant_violation": bool_values[1],
                                "rant_violation": bool_values[2],
                                "overall_compliant": not any(bool_values)
                            })
                        else:
                            results.append(self._create_fallback_result())
                    else:
                        results.append(self._create_fallback_result())
                        
            except Exception as e:
                results.append(self._create_fallback_result())
        
        return results
    
    def _extract_booleans_from_line(self, line: str) -> List[bool]:
        """Extract three boolean values from a line"""
        line = line.lower()
        
        # Look for true/false patterns
        bool_pattern = r'\b(true|false)\b'
        matches = re.findall(bool_pattern, line)
        
        if len(matches) >= 3:
            return [match == 'true' for match in matches[:3]]
        
        # Alternative: look for yes/no patterns
        yesno_pattern = r'\b(yes|no)\b'
        matches = re.findall(yesno_pattern, line)
        
        if len(matches) >= 3:
            return [match == 'yes' for match in matches[:3]]
        
        return None
    
    def _create_fallback_result(self) -> Dict:
        """Create fallback result when parsing fails"""
        return {
            "advertisement_violation": False,
            "irrelevant_violation": False,
            "rant_violation": False,
            "overall_compliant": True
        }
    
    def classify_batch(self, review_texts: List[str], max_retries: int = 2) -> List[Dict]:
        """Classify a batch of reviews"""
        
        prompt = self.create_batch_prompt(review_texts)
        
        for attempt in range(max_retries):
            try:
                response = self.generate_with_qwen_batch(prompt)
                results = self.parse_batch_response(response, len(review_texts))
                
                # Ensure we have results for all reviews
                while len(results) < len(review_texts):
                    results.append(self._create_fallback_result())
                
                return results[:len(review_texts)]
                
            except Exception as e:
                print(f"Batch error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return [self._create_fallback_result() for _ in review_texts]
                time.sleep(1)
        
        return [self._create_fallback_result() for _ in review_texts]
    
    def _create_short_review_response(self, review_text: str) -> Dict:
        """Create response for very short reviews"""
        return {
            'review_id': 'auto',
            'original_text': review_text,
            'overall_compliant': True,
            'advertisement_violation': False,
            'irrelevant_violation': False,
            'rant_violation': False
        }
    
    def classify_dataset(self, df: pd.DataFrame, text_column: str = 'text', 
                        batch_size: int = 8, delay: float = 0.3) -> pd.DataFrame:
        """Classify entire dataset with batch processing"""
        
        results = []
        total_reviews = len(df)
        
        print(f"Starting classification of {total_reviews} reviews...")
        print(f"Using device: {model.device}")
        print(f"Batch size: {batch_size}")
        
        # Process in batches
        for start_idx in tqdm(range(0, total_reviews, batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, total_reviews)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Prepare batch texts
            batch_texts = []
            batch_metadata = []
            
            for idx, row in batch_df.iterrows():
                review_text = str(row[text_column])
                
                if len(review_text.strip()) < 5:
                    batch_texts.append(review_text)
                    batch_metadata.append({'is_short': True, 'original_idx': idx})
                else:
                    batch_texts.append(review_text)
                    batch_metadata.append({'is_short': False, 'original_idx': idx})
            
            # Classify the batch
            batch_results = self.classify_batch(batch_texts)
            
            # Process results
            for i, (metadata, classification) in enumerate(zip(batch_metadata, batch_results)):
                result = {
                    'review_id': metadata['original_idx'],
                    'original_text': batch_texts[i],
                    **classification
                }
                
                # Override short reviews
                if metadata['is_short']:
                    result.update({
                        'advertisement_violation': False,
                        'irrelevant_violation': False,
                        'rant_violation': False,
                        'overall_compliant': True
                    })
                
                results.append(result)
            
            # Rate limiting and memory management
            time.sleep(delay)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original dataframe
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        return final_df

# Utility functions remain the same
def analyze_results(df: pd.DataFrame):
    """Analyze classification results"""
    
    print("=== CLASSIFICATION SUMMARY ===")
    print(f"Total reviews: {len(df)}")
    print(f"Compliant reviews: {df['overall_compliant'].sum()}")
    print(f"Non-compliant reviews: {(~df['overall_compliant']).sum()}")
    
    # Policy-specific violations
    violation_columns = ['advertisement_violation', 'irrelevant_violation', 'rant_violation']
    
    print("\n=== POLICY VIOLATIONS ===")
    for col in violation_columns:
        policy_name = col.replace('_violation', '').replace('_', ' ').title()
        violations = df[col].sum()
        print(f"{policy_name}: {violations} violations ({violations/len(df)*100:.1f}%)")
    
    return df

def save_results(df: pd.DataFrame, filename: str = 'classified_reviews.csv'):
    """Save results with proper formatting"""
    
    violation_columns = ['advertisement_violation', 'irrelevant_violation', 'rant_violation']
    
    df['violation_count'] = 0
    for col in violation_columns:
        df[col] = df[col].fillna(False)
        df['violation_count'] += df[col].astype(int)
    
    df.to_csv(filename, index=False)
    
    summary_cols = ['review_id', 'overall_compliant', 'violation_count'] + violation_columns
    summary_df = df[summary_cols]
    summary_df.to_csv(filename.replace('.csv', '_summary.csv'), index=False)
    
    print(f"Results saved to {filename}")
    print(f"Summary saved to {filename.replace('.csv', '_summary.csv')}")

def test_single_review():
    """Test the classifier with a single review"""
    classifier = ReviewPolicyClassifier()
    
    test_reviews = [
        "Amazing restaurant! The food was delicious and service was excellent.",
        "Visit my website www.bestdeals.com for great discounts!",
        "I love my new car, but this place has terrible parking.",
        "Never been here but I heard the food is awful."
    ]
    
    for i, review in enumerate(test_reviews):
        print(f"\n=== Test Review {i+1} ===")
        print(f"Review: {review}")
        result = classifier.classify_review(review)
        print(f"Result: {json.dumps(result, indent=2)}")
        time.sleep(1)

def test_batch_processing():
    """Test the classifier with batch processing"""
    classifier = ReviewPolicyClassifier()
    
    test_reviews = [
        "Amazing restaurant! The food was delicious and service was excellent.",
        "Visit my website www.bestdeals.com for great discounts!",
        "I love my new car, but this place has terrible parking.",
        "Never been here but I heard the food is awful."
    ]
    
    print("Testing batch processing...")
    results = classifier.classify_batch(test_reviews)
    
    for review, result in zip(test_reviews, results):
        print(f"\nReview: {review}")
        print(f"Advertisement: {result['advertisement_violation']}")
        print(f"Irrelevant: {result['irrelevant_violation']}")
        print(f"Rant: {result['rant_violation']}")
        print(f"Compliant: {result['overall_compliant']}")

def main():
    """Main function to run the classifier"""
    
    # Test both single and batch processing
    print("Testing single review processing...")
    # test_single_review()
    
    print("\nTesting batch processing...")
    # test_batch_processing()
    
    # Load your dataset
    df = pd.read_json('data/review-Alabama_10.json', lines=True)
    
    # Clean the dataset
    df = df.dropna(subset=['text', 'pics'])
    print(f"Loaded dataset with {len(df)} reviews")
    
    df1 = df.drop(columns=['pics','resp'])
    duplicate_indices = df1.index[df1.duplicated(keep="first")]
    
    df_clean = df.drop(index=duplicate_indices).reset_index(drop=True)
    df_clean["id"] = df_clean.index
    
    # Initialize classifier
    classifier = ReviewPolicyClassifier()
    
    # Process with optimized batch size
    sample_df = df_clean
    print("Processing with batch optimization...")
    start_time = time.time()
    classified_sample = classifier.classify_dataset(sample_df, text_column='text', batch_size=16)
    end_time = time.time()
    
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per review: {(end_time - start_time) / len(sample_df):.3f} seconds")
    
    # Analyze results
    analyze_results(classified_sample)
    save_results(classified_sample, 'optimized_classified_reviews.csv')

if __name__ == "__main__":
    main()