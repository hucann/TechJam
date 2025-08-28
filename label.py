import pandas as pd
import openai
from openai import OpenAI
import json
import time
from typing import Dict, List
import re

# Initialize OpenAI client
client = OpenAI(api_key='your-api-key-here')  # Replace with your actual API key

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
        
        prompt = """You are a review policy classifier. Analyze reviews and determine if they violate any of these policies:

1. **No Advertisement**: Reviews should not contain promotional content, links, or advertisements.
2. **No Irrelevant Content**: Reviews must be about the location/business, not unrelated topics.
3. **No Rant Without Visit**: Rants or complaints must come from actual visitors (check for indicators of actual visits).

Here are examples for each policy:

## No Advertisement Examples:
"""
        
        # Add examples for each policy
        for policy, examples in self.policy_examples.items():
            prompt += f"\n## {policy} Examples:\n"
            for i, example in enumerate(examples, 1):
                status = "VIOLATION" if example["violation"] else "NO VIOLATION"
                prompt += f"Example {i}:\n"
                prompt += f"Review: \"{example['review']}\"\n"
                prompt += f"Result: {status} - {example['reason']}\n\n"
        
        prompt += f"""
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
}}
"""
        return prompt
    
    def classify_review(self, review_text: str, max_retries: int = 3) -> Dict:
        """Classify a single review using GPT-4o"""
        
        prompt = self.create_few_shot_prompt(review_text)
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert content moderator specializing in review policy compliance."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                result = response.choices[0].message.content.strip()
                
                # Try to parse JSON response
                if result.startswith('```json'):
                    result = result.replace('```json', '').replace('```', '').strip()
                
                parsed_result = json.loads(result)
                return parsed_result
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self._create_error_response(f"JSON parsing failed: {e}")
                    
            except Exception as e:
                print(f"API error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self._create_error_response(f"API error: {e}")
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
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
                        batch_size: int = 10, delay: float = 1.0) -> pd.DataFrame:
        """Classify entire dataset with rate limiting"""
        
        results = []
        total_reviews = len(df)
        
        print(f"Starting classification of {total_reviews} reviews...")
        
        for idx, row in df.iterrows():
            review_text = str(row[text_column])
            
            print(f"Processing review {idx + 1}/{total_reviews}")
            
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
            
            # Rate limiting
            if (idx + 1) % batch_size == 0:
                print(f"Completed batch. Waiting {delay} seconds...")
                time.sleep(delay)
            
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original dataframe
        final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        return final_df

# Usage Example
def main():
    # Load your dataset
    df = pd.read_json('data/review-Alabama_10.json', lines=True)
    
    # Clean the dataset (remove None values)
    df = df.dropna(subset=['text'])  # Adjust column name as needed
    
    print(f"Loaded dataset with {len(df)} reviews")
    
    # Initialize classifier
    classifier = ReviewPolicyClassifier()
    
    # Test on a small sample first
    sample_df = df.head(5)  # Test with first 5 reviews
    
    print("Testing on sample...")
    classified_sample = classifier.classify_dataset(
        sample_df, 
        text_column='text',  # Adjust column name as needed
        batch_size=2,
        delay=1.0
    )
    
    print("\nSample Results:")
    print(classified_sample[['text', 'overall_compliant', 'summary']].head())
    
    # Uncomment to run on full dataset
    # print("Running on full dataset...")
    # classified_df = classifier.classify_dataset(df, text_column='text')
    # classified_df.to_csv('classified_reviews.csv', index=False)
    
    return classified_sample

# Additional utility functions
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

if __name__ == "__main__":
    # Set your OpenAI API key
    # client.api_key = "your-api-key-here"  # Replace with your actual key
    
    classified_df = main()
    analyze_results(classified_df)
    save_results(classified_df)