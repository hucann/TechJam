from __future__ import annotations
import json, re, math
import torch
import pandas as pd
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm  # For progress bars

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Initialize Qwen model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",              # Use GPU if available, otherwise CPU
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
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
                    "violation": True,
                    "reason": "Contains promotional link"
                },
                {
                    "review": "Amazing food and great service. Highly recommend this place!",
                    "violation": False,
                    "reason": "Genuine recommendation without promotional content"
                }
            ],
            "No Irrelevant Content": [
                {
                    "review": "I love my new phone, but this place is too noisy.",
                    "violation": True,
                    "reason": "Review contains irrelevant content about phone"
                },
                {
                    "review": "The atmosphere was perfect for a quiet dinner. Food was delicious.",
                    "violation": False,
                    "reason": "Review focuses on the restaurant experience"
                }
            ],
            "No Rant Without Visit": [
                {
                    "review": "Never been here, but I heard it's terrible.",
                    "violation": True,
                    "reason": "Reviewer explicitly states they haven't visited"
                },
                {
                    "review": "Visited last week and was disappointed with the slow service.",
                    "violation": False,
                    "reason": "Reviewer indicates they actually visited"
                }
            ]
        }
    
    def create_few_shot_prompt(self, review_text: str) -> str:
        """Create a few-shot prompt for policy classification"""
        
        prompt = """<|im_start|>system
You are a review policy classifier. Analyze reviews and determine if they violate any of these policies:

1. **No Advertisement**: Reviews should not contain promotional content, links, or advertisements.
2. **No Irrelevant Content**: Reviews must be about the location/business, not unrelated topics.
3. **No Rant Without Visit**: Rants or complaints must come from actual visitors (check for indicators of actual visits).

Here are examples for each policy:
<|im_end|>
"""
        
        # Add examples for each policy
        for policy, examples in self.policy_examples.items():
            prompt += f"\n<|im_start|>user\n## {policy} Examples:<|im_end|>\n"
            for i, example in enumerate(examples, 1):
                status = "VIOLATION" if example["violation"] else "NO VIOLATION"
                prompt += f"<|im_start|>user\nExample {i}:\n"
                prompt += f"Review: \"{example['review']}\"\n"
                prompt += f"Result: {status} - {example['reason']}\n\n<|im_end|>\n"
        
        prompt += f"""
<|im_start|>user
Now analyze this review:
Review: "{review_text}"

Please respond in the following JSON format:
{{
    "violations": [
        {{
            "policy": "policy_name",
            "violated": true/false,
            "confidence": 0.0-1.0,
            "reason": "explanation"
        }}
    ],
    "overall_compliant": true/false,
    "summary": "brief summary of findings"
}}<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def generate_with_qwen(self, prompt: str, max_new_tokens: int = 500) -> str:
        """Generate response using Qwen model"""
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        
        # Move to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract only the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        return response.strip()
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from model response"""
        
        # Try to find JSON in the response
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, response, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                json_str = json_str.replace("'", '"').replace("True", "true").replace("False", "false")
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        # If JSON extraction fails, create a fallback response
        return self._create_fallback_response(response)
    
    def _create_fallback_response(self, response: str) -> Dict:
        """Create fallback response when JSON parsing fails"""
        return {
            "violations": [
                {
                    "policy": "Error",
                    "violated": False,
                    "confidence": 0.0,
                    "reason": f"JSON parsing failed. Raw response: {response[:200]}..."
                }
            ],
            "overall_compliant": True,
            "summary": "Error in processing - could not parse model response"
        }
    
    def classify_review(self, review_text: str, max_retries: int = 2) -> Dict:
        """Classify a single review using Qwen model"""
        
        prompt = self.create_few_shot_prompt(review_text)
        
        for attempt in range(max_retries):
            try:
                # Generate response
                response = self.generate_with_qwen(prompt)
                
                # Try to parse JSON response
                parsed_result = self.extract_json_from_response(response)
                return parsed_result
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self._create_error_response(f"Processing error: {e}")
                
                # Wait before retry
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
    
    def classify_dataset(self, df: pd.DataFrame, text_column: str = 'text', 
                        batch_size: int = 1, delay: float = 0.5) -> pd.DataFrame:
        """Classify entire dataset with rate limiting"""
        
        results = []
        total_reviews = len(df)
        
        print(f"Starting classification of {total_reviews} reviews...")
        print(f"Using device: {model.device}")
        
        for idx, row in tqdm(df.iterrows(), total=total_reviews, desc="Processing reviews"):
            review_text = str(row[text_column])
            
            # Skip very short or empty reviews
            if len(review_text.strip()) < 5:
                result = self._create_short_review_response(review_text)
                results.append(result)
                continue
            
            # Classify the review
            classification = self.classify_review(review_text)
            
            # Extract key information
            result = {
                'review_id': idx,
                'original_text': review_text,
                'overall_compliant': classification.get('overall_compliant', True),
                'summary': classification.get('summary', ''),
            }
            
            # Add violation details
            violations = classification.get('violations', [])
            for violation in violations:
                policy_name = violation.get('policy', 'Unknown').replace(' ', '_').lower()
                result[f'{policy_name}_violation'] = violation.get('violated', False)
                result[f'{policy_name}_confidence'] = violation.get('confidence', 0.0)
                result[f'{policy_name}_reason'] = violation.get('reason', '')
            
            results.append(result)
            
            # Rate limiting and memory management
            if (idx + 1) % batch_size == 0:
                time.sleep(delay)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original dataframe
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        return final_df
    
    def _create_short_review_response(self, review_text: str) -> Dict:
        """Create response for very short reviews"""
        return {
            'review_id': 'auto',
            'original_text': review_text,
            'overall_compliant': True,
            'summary': 'Short review - automatically compliant',
            'no_advertisement_violation': False,
            'no_advertisement_confidence': 1.0,
            'no_advertisement_reason': 'Short review',
            'no_irrelevant_content_violation': False,
            'no_irrelevant_content_confidence': 1.0,
            'no_irrelevant_content_reason': 'Short review',
            'no_rant_without_visit_violation': False,
            'no_rant_without_visit_confidence': 1.0,
            'no_rant_without_visit_reason': 'Short review'
        }

# Utility functions
def analyze_results(df: pd.DataFrame):
    """Analyze classification results"""
    
    print("=== CLASSIFICATION SUMMARY ===")
    print(f"Total reviews: {len(df)}")
    print(f"Compliant reviews: {df['overall_compliant'].sum()}")
    print(f"Non-compliant reviews: {(~df['overall_compliant']).sum()}")
    
    # Policy-specific violations
    violation_columns = [col for col in df.columns if col.endswith('_violation')]
    
    print("\n=== POLICY VIOLATIONS ===")
    for col in violation_columns:
        policy_name = col.replace('_violation', '').replace('_', ' ').title()
        violations = df[col].sum()
        print(f"{policy_name}: {violations} violations ({violations/len(df)*100:.1f}%)")
    
    return df

def save_results(df: pd.DataFrame, filename: str = 'classified_reviews.csv'):
    """Save results with proper formatting"""
    
    # Create summary columns
    df['violation_count'] = 0
    violation_columns = [col for col in df.columns if col.endswith('_violation')]
    
    for col in violation_columns:
        df['violation_count'] += df[col].astype(int)
    
    # Save full results
    df.to_csv(filename, index=False)
    
    # Save summary
    summary_cols = ['review_id', 'overall_compliant', 'violation_count', 'summary'] + violation_columns
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

def main():
    """Main function to run the classifier"""
    
    # Test with single reviews first
    print("Testing with sample reviews...")
    test_single_review()
    
    # Load your dataset (uncomment when ready)
    # df = pd.read_json('data/review-Alabama_10.json', lines=True)
    
    # # Clean the dataset
    # df = df.dropna(subset=['text', 'pics'])
    # print(f"Loaded dataset with {len(df)} reviews")
    
    ## since pics and resp columns consist of list and dict, will need to remove to check the duplicate before continue
    # df1 = df.drop(columns=['pics','resp'])
    # duplicate_indices = df1.index[df1.duplicated(keep="first")]
    
    ## clean dataframe where it remove the duplicated rows according to ['user_id', 'name', 'time', 'rating', 'text', 'gmap_id']
    # df_clean = df.drop(index=duplicate_indices).reset_index(drop=True)
    # df_clean["id"] = df_clean.index
    
    # # Initialize classifier
    # classifier = ReviewPolicyClassifier()
    
    # # Run on a small sample first
    # sample_df = df_clean.head(3)
    # print("Processing sample...")
    # classified_sample = classifier.classify_dataset(sample_df, text_column='text')
    
    # print("\nSample Results:")
    # print(classified_sample[['text', 'overall_compliant', 'summary']].head())
    
    # # Analyze results
    # analyze_results(classified_sample)
    # save_results(classified_sample, 'sample_classified_reviews.csv')

if __name__ == "__main__":
    main()
