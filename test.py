from summarize import TextSummarizer
from rouge import Rouge

examples = [
    {
        'text': 
            '''The quick brown fox jumps over the lazy dog in a graceful leap. This sentence is often 
            used to demonstrate the use of every letter in the English alphabet. It is a popular pangram 
            and is commonly used for testing typewriters and keyboards around the world. The phrase has
            become a well-known example in the study of typography and font design, allowing designers to
            see how each letter appears in context. Over time, it has also been used in educational 
            settings to teach students about the alphabet and sentence structure. Many writers and 
            educators still reference this example because it compactly shows letterform coverage and 
            readability. Typography experts have analyzed this sentence countless times to understand 
            optimal letter spacing. Students learning graphic design often start their typography journey 
            with this classic example. The sentence's popularity has spawned numerous variations and 
            adaptations across different languages. Modern digital fonts are still tested using this 
            timeless phrase to ensure consistent rendering.'''
        ,
        'reference': 
            '''The quick brown fox jumps over the lazy dog is a popular pangram that uses every letter 
            in the English alphabet. This sentence is commonly used for testing typewriters and keyboards, 
            and has become a well-known example in typography and font design studies. It allows designers 
            to see how each letter appears in context and is used in educational settings to teach students 
            about alphabet and sentence structure. Typography experts analyze this sentence to understand 
            optimal letter spacing, and students learning graphic design often start with this classic example. 
            The sentence has spawned numerous variations across different languages and modern digital fonts 
            are still tested using this phrase.'''
    },
    {
        'text': 
            '''Artificial intelligence is rapidly transforming the world in unprecedented ways. From 
            healthcare to transportation, AI technologies are being integrated into various industries at 
            an accelerating pace. These technological advancements are leading to increased efficiency and 
            new possibilities for innovation across sectors. Machine learning algorithms are becoming 
            increasingly sophisticated, enabling computers to perform tasks that once required human 
            intelligence. In healthcare, AI-powered diagnostic tools are improving patient outcomes and 
            helping doctors make more accurate diagnoses. Autonomous vehicles powered by AI are reshaping 
            the future of mobility and transportation safety. Natural language processing is revolutionizing 
            how we interact with computers and digital assistants. As AI systems continue to evolve, they 
            are expected to create new opportunities and challenges across the global economy. The 
            integration of AI in manufacturing has led to smarter factories and more efficient production 
            processes. Ethical considerations and workforce impacts are driving important conversations 
            about responsible AI deployment. Researchers are actively working on making AI systems more 
            transparent and accountable. The potential impact of AI on society has become a central topic 
            in policy discussions worldwide.'''
        ,
        'reference': 
            '''Artificial intelligence is rapidly transforming the world by being integrated into various 
            industries including healthcare, transportation, and manufacturing at an accelerating pace. 
            Machine learning algorithms are becoming increasingly sophisticated, enabling computers to perform 
            tasks that once required human intelligence. In healthcare, AI-powered diagnostic tools are 
            improving patient outcomes, while autonomous vehicles are reshaping transportation safety. 
            Natural language processing is revolutionizing interactions with computers and digital assistants. 
            AI integration in manufacturing has led to smarter factories and more efficient production processes. 
            These technological advancements are creating new opportunities and challenges across the global 
            economy, while ethical considerations and workforce impacts are driving important conversations 
            about responsible AI deployment and the need for transparent and accountable AI systems.'''
    }
]

model_options = {
    'DistilBART (Recommended)': 'sshleifer/distilbart-cnn-12-6',
    'T5-Small (Fast)': 't5-small',
    'BART-Large (High Quality)': 'facebook/bart-large-cnn'
}
rouge = Rouge()

for name, model in model_options.items() :
    summarizer = TextSummarizer(model)
    print(f'\nModel: {name}\n')

    for example in examples:
        cleaned = summarizer.clean_text(example['text'])
        hypothesis = summarizer.summarize(cleaned)['summary']
        
        scores = rouge.get_scores(hypothesis, example['reference'])
        
        print(f'Generated Summary:\n{hypothesis}\n')
        print(f'ROUGE Scores:\n{scores}\n')