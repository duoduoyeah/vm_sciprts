from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_name = "Qwen/Qwen3-4B-Thinking-2507"

print("Loading model and tokenizer...")
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded successfully!\n")

def generate_response(prompt):
    """Generate a response for the given prompt."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    # Decode with special tokens visible
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=False).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=False).strip("\n")
    
    return thinking_content, content

def main():
    """Interactive chat loop."""
    print("=" * 60)
    print("Qwen3 TinyChat - Interactive CLI")
    print("=" * 60)
    print("Type your message (multiple lines supported).")
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
            print("\nGenerating response...\n")
            thinking, response = generate_response(user_input)
            
            # Display response
            if thinking:
                print(f"[Thinking]: {thinking}\n")
            print(f"Assistant: {response}\n")
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
