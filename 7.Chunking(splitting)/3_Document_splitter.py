from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
text = '''
        import pandas as pd

        # Load transaction data (assuming a CSV with 'Date', 'Amount', and 'Category' columns)
        def analyze_expenses(file_path, budget_limit):
            try:
                df = pd.read_csv(file_path)
                
                # Calculate total spending by category
                summary = df.groupby('Category')['Amount'].sum()
                print("Spending Summary by Category:\n", summary)

                # Check if 'Dining' exceeds the limit
                dining_spend = summary.get('Dining', 0)
                if dining_spend > budget_limit:
                    print(f"⚠️ Alert! You spent ${dining_spend:.2f} on dining, exceeding your ${budget_limit} limit.")
                else:
                    print(f"✅ You are within your dining budget. Total: ${dining_spend:.2f}")

            except FileNotFoundError:
                print("Error: The transaction file was not found.")

        # Example usage (Make sure 'transactions.csv' exists in your directory)
        analyze_expenses('transactions.csv', budget_limit=200)
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size= 200,
    chunk_overlap=0,
    language= Language.PYTHON
)

result= splitter.split_text(text)
print(result)
print(len(result))