import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Step 1: Web Scraping
def scrape_books(base_url, pages=1):
    book_data = []
    for page in range(1, pages + 1):
        page_url = f"{base_url}page-{page}.html" if page > 1 else f"{base_url}index.html"
        response = requests.get(page_url)
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}. Status code: {response.status_code}")
            continue
        soup = BeautifulSoup(response.content, "html.parser")
        books = soup.find_all("article", class_="product_pod")
        
        for book in books:
            title = book.h3.a["title"]
            price = book.find("p", class_="price_color").text.strip("£")
            availability = book.find("p", class_="instock availability").text.strip()
            rating = book.p["class"][1] if book.p and "class" in book.p.attrs else "Not Rated"
            
            book_data.append({
                "Title": title,
                "Price": float(price),
                "Availability": availability,
                "Rating": rating
            })
    
    if not book_data:
        print("No data scraped. Exiting.")
        exit()
    return pd.DataFrame(book_data)

# Step 2: Data Cleaning
def clean_data(df):
    if "Rating" in df.columns:
        rating_mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
        df["Rating"] = df["Rating"].map(rating_mapping).fillna(0)  # Map ratings to numbers
    
    if "Price" in df.columns:
        df["Price"] = (df["Price"] - df["Price"].mean()) / df["Price"].std()  # Normalize Price
    
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Step 3: Exploratory Data Analysis (EDA)
def perform_eda(df):
    if "Price" in df.columns:
        sns.histplot(df["Price"], kde=True)
        plt.title("Price Distribution")
        plt.show()
    
    if "Rating" in df.columns:
        sns.countplot(x="Rating", data=df)
        plt.title("Rating Count")
        plt.show()
    
    if "Availability" in df.columns and "Price" in df.columns:
        sns.boxplot(x="Availability", y="Price", data=df)
        plt.title("Price by Availability")
        plt.show()
# Step 4: Machine Learning Models
def train_book_classification(df):
    df["Category"] = ["Travel"] * len(df)  # Placeholder for category
    X = df[["Price", "Rating"]]
    y = df["Category"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Book Classification:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Updated Price Prediction Function
def train_price_prediction(df):
    if "Rating" in df.columns and "Price" in df.columns:
        # Avoid using Price as a predictor, use only Rating
        X = df[["Rating"]]  # Removed "Price" from features
        y = df["Price"]
        
        # Test train split with better dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        # Manually calculate RMSE
        mse_value = mean_squared_error(y_test, y_pred)
        rmse = mse_value**0.5
        
        print("Price Prediction:")
        print("RMSE:", rmse)
        print("R² Score:", r2_score(y_test, y_pred))


# Main Execution
if _name_ == "_main_":
    print("Scraping data...")
    base_url = "https://books.toscrape.com/catalogue/category/books/travel_2/"
    books_df = scrape_books(base_url, pages=2)  # Adjust 'pages' as needed
    print("Scraping completed. Saving raw data...")
    books_df.to_csv("books_data.csv", index=False)
    
    print("Cleaning data...")
    cleaned_df = clean_data(books_df)
    cleaned_df.to_csv("cleaned_books_data.csv", index=False)
    print("Data cleaned and saved.")
    
    print("Performing EDA...")
    perform_eda(cleaned_df)
    
    print("Training models...")
    train_book_classification(cleaned_df)
    train_price_prediction(cleaned_df)
    print("All tasks completed!")