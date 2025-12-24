from encodings.aliases import aliases
from pymongo import MongoClient
import re
from tx_sandbox import clean_example_text, extract_amount, remove_accents, parse_direction, normalize_category
from typing import Dict



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

def modelize(raw: str, userId: str = None) -> Dict[str, object]:
    amount    = extract_amount(raw)
    direction = parse_direction(raw)

    # # A) Rules-first
    # hit = classify_by_rules(raw)
    # if hit:
    #     cid, cname = hit
    #     return {
    #         "raw": raw, "amount": amount, "currency": "VND", "direction": direction,
    #         "categoryId": cid, "category_name": cname,
    #         "confidence": 0.99, "decision_source": "RULE"
    #     }
 
    # B) Embeddings (optional)
    # elif use_embeddings:
    cid = categorizeItem(userId, clean_example_text(raw))
    return {
        "raw": raw, "amount": amount, "direction": direction,
        "categoryId": cid,
    }

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
    print(sims)
    i = int(np.argmax(sims)); score = float(sims[i])
    return (user_categories[i]["categoryId"] if score>=0.8 else "Other/Review")
    

# Add a new example sentence to a user's category
def addCategoryExample(userId: str, categoryName: str, example: str):
    # remove example from other categories first
    accentfree_name = remove_accents(categoryName)
    normalized_name = normalize_category(accentfree_name)
    categoryId = normalized_name.replace(" ", "_")
    accentfree_example = remove_accents(example)
    clean_example = clean_example_text(accentfree_example)
    print ("Cleaned example:", clean_example)
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
            "text": clean_example,
            "embedding": model.encode([remove_accents(clean_example)], normalize_embeddings=True)[0].tolist()
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
    accentfree_example = remove_accents(example)
    clean_example = clean_example_text(accentfree_example)            
    result = category_collection.update_one(
        {"userId": userId, "categoryId": categoryId},
        {"$push": {"exampleList": {
            "text": clean_example,
            "embedding": model.encode([clean_example], normalize_embeddings=True)[0].tolist()
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

def resetCentroids(userId: str, categoryId = None):
    query = {"userId": userId}
    if categoryId:
        query["categoryId"] = categoryId
    user_cats = category_collection.find(query)
    for cat in user_cats:
        ex_list = cat.get("exampleList", [])
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
    return "Centroids reset."

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