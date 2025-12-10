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
    print("Type your message and press Enter to chat.")
    print("Type 'exit', 'quit', or 'q' to end the conversation.")
    print("=" * 60)
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            # Skip empty inputs
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
