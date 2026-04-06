#============================================================
#   E-COMMERCE CUSTOMER BEHAVIOUR ANALYSIS
#   HOW TO RUN:
#   1. Put this file and ecommerce_data.csv in SAME folder
#   2. Open that folder in VS Code
#   3. Run: python analysis.py
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

# ── FIX 1: Auto-set working directory to this file's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Working directory:", os.getcwd())

# ── FIX 2: Auto-install any missing libraries ────────────────
import subprocess, sys
required = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "openpyxl"]
for pkg in required:
    try:
        __import__(pkg if pkg != "scikit-learn" else "sklearn")
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


# ============================================================
# STEP 1 — LOAD & CLEAN DATA
# ============================================================
print("\n" + "="*60)
print("  STEP 1: DATA CLEANING")
print("="*60)

# ── FIX 3: Check CSV exists with helpful message ─────────────
csv_file = "ecommerce_data.csv"
if not os.path.exists(csv_file):
    print(f"\n  ERROR: '{csv_file}' not found!")
    print(f"  Please put ecommerce_data.csv here: {os.getcwd()}")
    sys.exit()

df = pd.read_csv(csv_file, encoding="ISO-8859-1")

print(f"Original shape      : {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# 1a. Drop missing Customer ID and Product Name
df.dropna(subset=["Customer ID"], inplace=True)
df.dropna(subset=["Product Name"], inplace=True)

# 1b. Remove cancelled orders (Invoice No starts with 'C')
df = df[~df["Invoice No"].astype(str).str.startswith("C")]

# 1c. Remove negative or zero Quantity and Price
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# 1d. Convert Invoice Date to datetime
df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], infer_datetime_format=True)

# 1e. Extract time features
df["Year"]        = df["Invoice Date"].dt.year
df["Month"]       = df["Invoice Date"].dt.month
df["Month_Name"]  = df["Invoice Date"].dt.strftime("%b")
df["Day_of_Week"] = df["Invoice Date"].dt.day_name()

# 1f. Create TotalPrice column
df["TotalPrice"] = df["Quantity"] * df["Price"]

print(f"\nCleaned shape       : {df.shape}")
print(f"\nSample rows:")
print(df[["Customer ID","Invoice No","Product Name",
          "Quantity","Price","TotalPrice","Country"]].head(5).to_string())


# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "="*60)
print("  STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

# 2a. Top 10 products by revenue
top_products = (
    df.groupby("Product Name")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_products.columns = ["Product", "Revenue"]
print("\nTop 10 Products by Revenue:")
print(top_products.to_string(index=False))

# 2b. Monthly sales trend
monthly_sales = (
    df.groupby(["Year", "Month"])["TotalPrice"]
    .sum()
    .reset_index()
)
monthly_sales["Period"] = (
    monthly_sales["Year"].astype(str) + "-" +
    monthly_sales["Month"].astype(str).str.zfill(2)
)
monthly_sales = monthly_sales.sort_values("Period")
print("\nMonthly Sales Trend:")
print(monthly_sales[["Period", "TotalPrice"]].to_string(index=False))

# 2c. Country-wise sales
country_sales = (
    df.groupby("Country")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
print("\nTop 10 Countries by Revenue:")
print(country_sales.to_string(index=False))

# 2d. New vs Repeat customers
customer_orders  = df.groupby("Customer ID")["Invoice No"].nunique()
new_customers    = (customer_orders == 1).sum()
repeat_customers = (customer_orders  > 1).sum()
total_customers  = len(customer_orders)
print(f"\nNew customers    : {new_customers}  ({new_customers/total_customers*100:.1f}%)")
print(f"Repeat customers : {repeat_customers} ({repeat_customers/total_customers*100:.1f}%)")


# ============================================================
# STEP 3 — RFM CUSTOMER SEGMENTATION
# ============================================================
print("\n" + "="*60)
print("  STEP 3: RFM ANALYSIS")
print("="*60)

reference_date = df["Invoice Date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg(
    Recency   = ("Invoice Date", lambda x: (reference_date - x.max()).days),
    Frequency = ("Invoice No",   "nunique"),
    Monetary  = ("TotalPrice",   "sum")
).reset_index()

# ── FIX 4: duplicates="drop" prevents qcut crash ────────────
rfm["R_Score"] = pd.qcut(rfm["Recency"],
                          q=4, labels=[4,3,2,1],
                          duplicates="drop").astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),
                          q=4, labels=[1,2,3,4],
                          duplicates="drop").astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"],
                          q=4, labels=[1,2,3,4],
                          duplicates="drop").astype(int)

rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

def assign_segment(score):
    if score >= 10:  return "Premium"
    elif score >= 7: return "Regular"
    elif score >= 5: return "At Risk"
    else:            return "Lost"

rfm["Segment"] = rfm["RFM_Score"].apply(assign_segment)

print("\nSegment Distribution:")
print(rfm["Segment"].value_counts().to_string())
print("\nAverage RFM by Segment:")
print(rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(2).to_string())


# ============================================================
# STEP 4 — VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("  STEP 4: SAVING CHARTS")
print("="*60)

# Chart 1: Top 10 Products
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_products, y="Product", x="Revenue",
            palette="Blues_r", ax=ax)
ax.set_title("Top 10 Products by Revenue", fontsize=14, fontweight="bold")
ax.set_xlabel("Total Revenue")
ax.set_ylabel("")
for p in ax.patches:
    ax.annotate(f"{p.get_width():,.0f}",
                (p.get_width(), p.get_y() + p.get_height()/2),
                ha="left", va="center", fontsize=8, color="gray")
plt.tight_layout()
plt.savefig("top_products.png")
plt.close()
print("  Saved → top_products.png")

# Chart 2: Monthly Sales Trend
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly_sales["Period"], monthly_sales["TotalPrice"],
        marker="o", color="#1f77b4", linewidth=2, markersize=5)
ax.fill_between(monthly_sales["Period"], monthly_sales["TotalPrice"],
                alpha=0.15, color="#1f77b4")
ax.set_title("Monthly Sales Trend", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("monthly_trend.png")
plt.close()
print("  Saved → monthly_trend.png")

# Chart 3: Country-wise Sales
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=country_sales, x="Country", y="TotalPrice",
            palette="viridis", ax=ax)
ax.set_title("Top 10 Countries by Revenue", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Revenue")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("country_sales.png")
plt.close()
print("  Saved → country_sales.png")

# Chart 4: Customer Segments Donut
segment_counts = rfm["Segment"].value_counts()
colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(segment_counts, labels=segment_counts.index,
       autopct="%1.1f%%", colors=colors[:len(segment_counts)],
       wedgeprops=dict(width=0.55), startangle=140)
ax.set_title("Customer Segments (RFM)", fontsize=14, fontweight="bold")
plt.savefig("segments.png")
plt.close()
print("  Saved → segments.png")

# Chart 5: RFM Heatmap
rfm_pivot = rfm.groupby(["R_Score","F_Score"])["Monetary"].mean().unstack()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(rfm_pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, ax=ax)
ax.set_title("Avg Revenue by Recency x Frequency Score",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Frequency Score (1=low, 4=high)")
ax.set_ylabel("Recency Score (1=old, 4=recent)")
plt.tight_layout()
plt.savefig("rfm_heatmap.png")
plt.close()
print("  Saved → rfm_heatmap.png")

# Chart 6: New vs Repeat Customers
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["New Customers", "Repeat Customers"],
       [new_customers, repeat_customers],
       color=["#64B5F6", "#1565C0"], width=0.5)
ax.set_title("New vs Repeat Customers", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
for i, v in enumerate([new_customers, repeat_customers]):
    ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("new_vs_repeat.png")
plt.close()
print("  Saved → new_vs_repeat.png")


# ============================================================
# STEP 5 — BUSINESS INSIGHTS REPORT
# ============================================================
print("\n" + "="*60)
print("  STEP 5: BUSINESS INSIGHTS")
print("="*60)

total_revenue   = df["TotalPrice"].sum()
total_orders    = df["Invoice No"].nunique()
avg_order_value = total_revenue / total_orders
best_month      = monthly_sales.loc[monthly_sales["TotalPrice"].idxmax(), "Period"]
top_country     = country_sales.iloc[0]["Country"]
top_product     = top_products.iloc[0]["Product"]
premium_pct     = (rfm["Segment"] == "Premium").mean() * 100
lost_pct        = (rfm["Segment"] == "Lost").mean()    * 100

print(f"""
  Revenue Summary
  ─────────────────────────────────────────────
  Total Revenue     : {total_revenue:>12,.2f}
  Total Orders      : {total_orders:>12,}
  Unique Customers  : {total_customers:>12,}
  Avg Order Value   : {avg_order_value:>12,.2f}

  Top Performers
  ─────────────────────────────────────────────
  Best Month        : {best_month}
  Top Country       : {top_country}
  Top Product       : {top_product[:45]}

  Customer Segments
  ─────────────────────────────────────────────
  Premium customers : {premium_pct:.1f}%
  Lost customers    : {lost_pct:.1f}%

  Recommendations
  ─────────────────────────────────────────────
  1. Run win-back campaigns for 'Lost' customers
  2. Offer loyalty rewards to 'Premium' customers
  3. Increase ad spend in top country : {top_country}
  4. Stock up on top product before peak month
  5. Upsell or discount offers for 'At Risk' customers
""")


# ============================================================
# STEP 6 — ADVANCED: RECOMMENDATIONS + CHURN PREDICTION
# ============================================================
print("="*60)
print("  STEP 6: ADVANCED ANALYSIS")
print("="*60)

# 6a. Product Recommendation
print("\n  [Recommendation System]")
customer_product = (
    df.groupby(["Customer ID", "Product Name"])["Quantity"]
    .sum().unstack(fill_value=0)
)
customer_product = (customer_product > 0).astype(int)

# ── FIX 5: guard against too few products ───────────────────
if customer_product.shape[1] > 5:
    product_corr = customer_product.corr(method="pearson")

    def recommend_products(product_name, top_n=5):
        if product_name not in product_corr.columns:
            return "Product not found."
        return (
            product_corr[product_name]
            .drop(product_name)
            .sort_values(ascending=False)
            .head(top_n)
        )

    sample_product = top_products["Product"].iloc[0]
    print(f"\n  Customers who bought:\n  '{sample_product[:50]}'\n  also liked:")
    print(recommend_products(sample_product).to_string())
else:
    print("  Not enough products for recommendations.")

# 6b. Churn Prediction
print("\n  [Churn Prediction]")
rfm["Churned"] = (rfm["Recency"] > 90).astype(int)
print(f"  Churned : {rfm['Churned'].sum()} | Active : {(rfm['Churned']==0).sum()}")

X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["Churned"]

# ── FIX 6: only train if both classes present ────────────────
if y.nunique() == 2:
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + classification_report(y_test, y_pred,
          target_names=["Active", "Churned"]))
    print(f"  ROC-AUC Score : {roc_auc_score(y_test, y_proba):.3f}")

    coefs = pd.Series(model.coef_[0],
                      index=["Recency", "Frequency", "Monetary"])
    print("\n  Feature Importance:")
    print(coefs.sort_values(ascending=False).to_string())
else:
    print("  Not enough variation in churn labels to train model.")


# ============================================================
print("\n" + "="*60)
print("  ALL DONE! Check your folder for PNG chart files.")
print("="*60 + "\n")