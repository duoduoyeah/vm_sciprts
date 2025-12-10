from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_name = 'roneneldan/TinyStories-33M'
tokenizer_name = "EleutherAI/gpt-neo-125M"

print("Loading model and tokenizer...")
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded successfully!\n")

def generate_response(prompt):
    """Generate a response for the given prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate completion
    output = model.generate(
        input_ids, 
        max_length=1000, 
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the completion
    output_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    return output_text

def main():
    """Interactive chat loop."""
    print("=" * 60)
    print("TinyStories Chat - Interactive CLI")
    print("=" * 60)
    print("Type your story prompt (multiple lines supported).")
    print("Press Enter 4 times (3 empty lines) to send your message.")
    print("Type 'exit', 'quit', or 'q' on a new line to end.")
    print("=" * 60)
    print()
    
    while True:
        try:
            # Get multi-line user input
            print("You: ", end="", flush=True)
            lines = []
            empty_count = 0
            while True:
                line = input()
                if line.strip() == "":
                    empty_count += 1
                    if empty_count >= 3 and lines:  # 3 empty lines after content means send
                        break
                    lines.append(line)
                else:
                    empty_count = 0
                    if line.strip().lower() in ['exit', 'quit', 'q'] and not lines:
                        print("\nGoodbye!")
                        return
                    lines.append(line)
            
            user_input = "\n".join(lines).strip()
            
            # Skip if somehow empty
            if not user_input:
                continue
            
            # Generate response
            print("\nGenerating story...\n")
            story = generate_response(user_input)
            
            # Display response
            print(f"Story: {story}\n")
            print("-" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

if __name__ == "__main__":
    main()
