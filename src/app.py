import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.answer import process_query

def main():
    print("\n" + "="*50)
    print("=== Internal Research RAG CLI ===")
    print("Type 'exit' to quit at any prompt.")
    print("="*50 + "\n")
    
    while True:
        query = input("Enter your query: ").strip()
        if query.lower() == 'exit':
            break
            
        country = input("Enter country (e.g. UAE, Australia, Thailand): ").strip()
        if country.lower() == 'exit':
            break
            
        category = input("Enter category (e.g. Ownership rules, Tax exposure...): ").strip()
        if category.lower() == 'exit':
            break
            
        print("\nSearching and generating strictly controlled answer...")
        answer, chunks = process_query(query, country, category)
        
        print("\n" + "="*60)
        print(answer)
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
