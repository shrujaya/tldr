from transformers import pipeline
import torch
import re
import time

class TextSummarizer :
    def __init__(self, model_name) :
        self.model_name = model_name
        self.device = self.get_device()
        self.summarizer = self.load_summarizer()

    def get_device(self) :
        if torch.backends.mps.is_available() :
            return 'mps'
        elif torch.cuda.is_available() :
            return 'cuda'
        else :
            return 'cpu'

    def load_summarizer(self) :
        try :
            return pipeline(
                task='summarization',
                model=self.model_name,
                device=self.device,
                torch_dtype=
                torch.float16 if self.device != 'cpu' else torch.float32
            )
        except Exception as e :
            print(f"Error loading model {self.model_name}: {e}")
            return pipeline(
                task='summarization',
                model='t5-small',
                device=self.device,
                torch_dtype=
                torch.float16 if self.device != 'cpu' else torch.float32
            )

    def clean_text(self, text) :
        if not text or not text.strip() :
            return "No content to summarize."
        
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)

        return text
    
    def validate_input(self, text) :
        word_count = len(text.split())
        char_count = len(text)

        validation = {
            'is_valid': True,
            'error_message': '', 
            'word_count': word_count,
            'char_count': char_count
        }

        if word_count < 50 :
            raise ValueError("Input text is too short. Summary may not be meaningful.")
        if word_count > 4000 :
            raise ValueError("Input text is too long. Consider breaking it into smaller chunks.")
        if char_count > 10000 :
            validation['is_valid'] = False
            raise ValueError("Text exceeds model's token limit. Please shorten the input.")
        
        return validation
    
    def summarize(self, text, max_length=150, min_length=30, do_sample=False, temperature=1.0) :
        start_time = time.time()
        
        cleaned_text = self.clean_text(text)
        valid = self.validate_input(cleaned_text)

        if not valid['is_valid'] :
            return {
                'success': False,   
                "error": valid['error_message']
            }

        try :
            summary = self.summarizer(
                cleaned_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                truncation=True
            )
            processing_time = time.time() - start_time
            summary_text = summary[0]['summary_text']

            original_word_count = len(cleaned_text.split())
            summary_word_count = len(summary_text.split())
            reduction_percent = round((1 - summary_word_count / original_word_count) * 100, 1)

            return {
                'success': True,
                'summary': summary_text,
                'metadata' : {
                    'original_word_count': original_word_count,
                    'summary_word_count': summary_word_count,
                    'reduction_percent': reduction_percent,
                    'processing_time': round(processing_time, 2),
                    'model_used': self.model_name,
                    'device': self.device
                }
            }
        
        except Exception as e :
            return {
                'success': False,   
                "error": f'Summarization failed: {str(e)}',
            }