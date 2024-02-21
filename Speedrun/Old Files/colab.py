def generate_custom_runtime_url(token, notebook_id):
    base_url = "https://colab.research.google.com/drive/"
    return f"{base_url}{notebook_id}?authuser=0#scrollTo={token}"

if __name__ == "__main__":
    token = input("Enter your runtime token: ")
    notebook_id = input("Enter your notebook ID: ")
    custom_url = generate_custom_runtime_url(token, notebook_id)
    print("Custom Runtime URL:", custom_url)
