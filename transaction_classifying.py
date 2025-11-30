from encodings.aliases import aliases
from pymongo import MongoClient
import re
import unicodedata

def normalize_category(name: str) -> str:
    name = name.lower().strip()
    name = name.replace("&", "and")
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\s*([,.;:])\s*", r"\1 ", name)
    return name.strip()

def remove_accents(input_str: str) -> str:
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])    

try:
    from .model_loader import load_model
except ImportError:
    from model_loader import load_model
import numpy as np

# Pick small for speed; upgrade to ...-base later if needed

model = load_model()
mongo_uri = "mongodb+srv://giathinh:qOI7BTIkAcOM8kTN@spring-base-project.dn9zo.mongodb.net/base-security?retryWrites=true&w=majority&appName=Spring-Base-Project"
client = MongoClient(mongo_uri)
db = client["base-security"]
category_collection = db["user_categories"] 

labels = [
  "Grocery","Food & Drinks","Transport"
]

# Seed a tiny bank of labeled descriptions (add as you collect corrections)
examples = [
  ("mua sữa và bánh mì", "Grocery"),
    ("ăn phở trưa", "Food & Drinks"),
    ("cơm tấm", "Food & Drinks"),
    ("grab đi trường", "Transport"),
    # ("phí xe bus", "Transport"),
    # ("đổ xăng", "Fuel"),
    # ("tiền điện evn", "Utilities"),
    # ("nạp ví momo", "Transfer"),
    # ("phí duy trì thẻ", "Fees"),
    # ("hoàn tiền đơn tiki", "Refund"),
    # ("coi phim netflix", "Entertainment"),
    # ("thể thao", "Entertainment"),
    # ("mua thuốc tây", "Healthcare"),
    # ("mua áo shopee", "Shopping"),
    # ("trả tiền nhà", "Rent"),
    # ("đầu tư ssi", "Investments"),
    # ("rút tiền atm", "Cash"),
]

# Build class centroids (you can recompute whenever you add examples)
X = model.encode([t for t,_ in examples], normalize_embeddings=True)
Y = np.array([labels.index(y) for _,y in examples])
centroids = []
for i,lab in enumerate(labels):
    vecs = X[Y==i]
    if len(vecs)==0:
        seed_texts = [labels[i], *aliases, labels[i].lower().replace("&","and")]
        c = model.encode(seed_texts, normalize_embeddings=True).mean(axis=0)
        centroids.append(c/np.linalg.norm(c))
    else:
        c = vecs.mean(axis=0)
        centroids.append(c/np.linalg.norm(c))
centroids = np.vstack(centroids)



# Initialize default categories for a new user
def initUserCategory(userId: str):
    existing = category_collection.find_one({"userId": userId})
    if existing:
        return "User categories already initialized."

    default_docs = []

    for cat in labels:
        # Collect example embeddings
        cat_examples = [
            {
                "text": remove_accents(ex_text),
                "embedding": model.encode([remove_accents(ex_text)], normalize_embeddings=True)[0].tolist()
            }
            for ex_text, ex_cat in examples if ex_cat == cat
        ]

        # Compute centroid (mean embedding)
        if len(cat_examples) > 0:
            centroid_vec = np.mean([e["embedding"] for e in cat_examples], axis=0)
            centroid_vec = centroid_vec.tolist()  # convert numpy → list for MongoDB
        else:
            centroid_vec = None

        # Create the category document
        default_docs.append({
            "userId": userId,
            "categoryId": normalize_category(cat).replace(" ", "_"),
            "categoryName": cat,
            "normalizedName": normalize_category(cat),
            "exampleList": cat_examples,
            "centroid": centroid_vec
        })

    category_collection.insert_many(default_docs)
    return "Initialized user categories."


def addCustomCategory(userId: str, categoryId: str, categoryName: str):
    accentfree_name = remove_accents(categoryName)
    normalized_name = normalize_category(accentfree_name)
    # categoryId = normalized_name.replace(" ", "_")
    existing = category_collection.find_one({"userId": userId, "categoryId": categoryId})
    if existing:
        return "Category already exists."
    initial_example  = model.encode([accentfree_name], normalize_embeddings=True)[0].tolist()
    new_category = {
        "userId": userId,
        "categoryId": categoryId,
        "categoryName": categoryName,
        "normalizedName": normalized_name,
        "exampleList": [],
        "centroid": initial_example
    }

    category_collection.insert_one(new_category)
    return "Custom category added."



def categorizeItem(userId: str, item: str):
    user_categories = list(category_collection.find({"userId": userId}))
    if not user_categories:
        return "User categories not initialized."
    v = model.encode([item], normalize_embeddings=True)[0]
    centroid_vec = [e["centroid"] for e in user_categories]
    sims = centroid_vec @ v
    i = int(np.argmax(sims)); score = float(sims[i])
    return (user_categories[i]["categoryId"] if score>=0.55 else "Other/Review")
    

# Add a new example sentence to a user's category
def addCategoryExample(userId: str, categoryName: str, example: str):
    # remove example from other categories first
    accentfree_name = remove_accents(categoryName)
    normalized_name = normalize_category(accentfree_name)
    categoryId = normalized_name.replace(" ", "_")
    all_user_cats = category_collection.find({"userId": userId})
    for cat in all_user_cats:
        if any(e["text"] == example for e in cat["exampleList"]):
            # Remove the example
            category_collection.update_one(
                {"userId": userId, "categoryId": cat["categoryId"]},
                {"$pull": {"exampleList": {"text": example}}}
            )

            # Fetch updated example list once
            updated_cat = category_collection.find_one(
                {"userId": userId, "categoryId": cat["categoryId"]}
            )
            ex_list = updated_cat.get("exampleList", [])

            # Recompute centroid only once
            if len(ex_list) > 0:
                new_centroid = np.mean(
                    [e["embedding"] for e in ex_list],
                    axis=0
                ).tolist()
            else:
                new_centroid = model.encode(
                    [remove_accents(cat["categoryName"])],
                    normalize_embeddings=True
                )[0].tolist()

            category_collection.update_one(
                {"userId": userId, "categoryId": cat["categoryId"]},
                {"$set": {"centroid": new_centroid}}
            )
    if not category_collection.find_one({"userId": userId, "categoryId": categoryId}):
        addCustomCategory(userId=userId, categoryName=categoryName)            
    result = category_collection.update_one(
        {"userId": userId, "categoryId": categoryId},
        {"$push": {"exampleList": {
            "text": example,
            "embedding": model.encode([remove_accents(example)], normalize_embeddings=True)[0].tolist()
        }}}
    )
    updated_cat = category_collection.find_one({"userId": userId, "categoryId": categoryId})
    ex_list = updated_cat.get("exampleList", [])
    new_centroid = np.mean([e["embedding"] for e in ex_list], axis=0).tolist()

    category_collection.update_one(
        {"userId": userId, "categoryId": categoryId},
        {"$set": {"centroid": new_centroid}}
    )
    if result.modified_count == 0:
        return "No matching category found."
    return "Example added."

def addCategoryExampleByCatgoryId(userId: str, categoryId: str, example: str):
    all_user_cats = category_collection.find({"userId": userId})
    for cat in all_user_cats:
        if any(e["text"] == example for e in cat["exampleList"]):
            # Remove the example
            category_collection.update_one(
                {"userId": userId, "categoryId": cat["categoryId"]},
                {"$pull": {"exampleList": {"text": example}}}
            )

            # Fetch updated example list once
            updated_cat = category_collection.find_one(
                {"userId": userId, "categoryId": cat["categoryId"]}
            )
            ex_list = updated_cat.get("exampleList", [])

            # Recompute centroid only once
            if len(ex_list) > 0:
                new_centroid = np.mean(
                    [e["embedding"] for e in ex_list],
                    axis=0
                ).tolist()
            else:
                new_centroid = model.encode(
                    [remove_accents(cat["categoryName"])],
                    normalize_embeddings=True
                )[0].tolist()

            category_collection.update_one(
                {"userId": userId, "categoryId": cat["categoryId"]},
                {"$set": {"centroid": new_centroid}}
            )
    if not category_collection.find_one({"userId": userId, "categoryId": categoryId}):
        addCustomCategory(userId=userId, categoryId=categoryId, categoryName=categoryId)            
    result = category_collection.update_one(
        {"userId": userId, "categoryId": categoryId},
        {"$push": {"exampleList": {
            "text": example,
            "embedding": model.encode([remove_accents(example)], normalize_embeddings=True)[0].tolist()
        }}}
    )
    updated_cat = category_collection.find_one({"userId": userId, "categoryId": categoryId})
    ex_list = updated_cat.get("exampleList", [])
    new_centroid = np.mean([e["embedding"] for e in ex_list], axis=0).tolist()

    category_collection.update_one(
        {"userId": userId, "categoryId": categoryId},
        {"$set": {"centroid": new_centroid}}
    )
    if result.modified_count == 0:
        return "No matching category found."
    return "Example added."

def deleteCategoryExample(userId: str, categoryId: str, example: str):
    result = category_collection.update_one(
        {"userId": userId, "categoryId": categoryId},
        {"$pull": {"exampleList": {"text": example}}}
    )
    if result.modified_count == 0:
        return "No matching category or example found."
    return "Example removed."

def deleteCategoryByUserId(userId: str):
    result = category_collection.delete_many(
        {"userId": userId}
    )
    if result.deleted_count == 0:
        return "No matching user categories found."
    return "User categories deleted."

def classify_desc(text: str, thres=0.55):
    v = model.encode([text], normalize_embeddings=True)[0]
    sims = centroids @ v
    i = int(np.argmax(sims)); score = float(sims[i])
    return (labels[i] if score>=thres else "Other/Review", score)