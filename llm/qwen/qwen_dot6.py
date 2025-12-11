from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_name = "Qwen/Qwen3-0.6B-Base"

print("Loading model and tokenizer...")
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("Model loaded successfully!\n")

def generate_response(conversation_history):
    """Generate a response for the given conversation history."""
    text = tokenizer.apply_chat_template(
        conversation_history,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # Decode the response
    content = tokenizer.decode(output_ids, skip_special_tokens=False).strip("\n")
    content_clean = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    
    return content, content_clean

def main():
    """Interactive chat loop."""
    print("=" * 60)
    print("Qwen3-0.6B Chat - Interactive CLI")
    print("=" * 60)
    print("Type your message and press Enter to send.")
    print("Type 'exit', 'quit', or 'q' to end.")
    print("Type 'clear' to reset conversation history.")
    print("=" * 60)
    print()
    
    # Initialize conversation history
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                return
            
            # Check for clear command
            if user_input.lower() == 'clear':
                conversation_history = []
                print("\nConversation history cleared!\n")
                print("-" * 60)
                print()
                continue
            
            # Skip if empty
            if not user_input:
                continue
            
            # Add user message to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Generate response
            print("\nGenerating response...\n")
            response, response_clean = generate_response(conversation_history)
            
            # Add assistant response to conversation history
            conversation_history.append({"role": "assistant", "content": response_clean})
            
            # Display response
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
