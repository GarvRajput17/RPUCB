import os
import random
from collections import defaultdict

def preprocess_citeulike(data_dir):
    random.seed(42)  # For reproducibility
    users_file = os.path.join(data_dir, 'users.dat')
    
    # 1. Read users.dat
    raw_user_items = []
    with open(users_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # Some datasets have the number of interactions as the first element
            if int(parts[0]) == len(parts) - 1:
                items = [int(p) for p in parts[1:]]
            else:
                items = [int(p) for p in parts]
            raw_user_items.append(items)
            
    # Construct list of (user_id, item_id)
    # The original order in the file is preserved as proxy for time.
    interactions = []
    for u, items in enumerate(raw_user_items):
        for pos, i in enumerate(items):
            interactions.append({'u': u, 'i': i, 'pos': pos})
            
    # 2. Filter users and items with < 5 interactions (iterative core filtering)
    while True:
        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        
        for interaction in interactions:
            user_counts[interaction['u']] += 1
            item_counts[interaction['i']] += 1
            
        filtered_interactions = [
            interaction for interaction in interactions
            if user_counts[interaction['u']] >= 5 and item_counts[interaction['i']] >= 5
        ]
        
        if len(filtered_interactions) == len(interactions):
            break
        interactions = filtered_interactions
        
    # Re-index
    unique_users = sorted(list(set(x['u'] for x in interactions)))
    unique_items = sorted(list(set(x['i'] for x in interactions)))
    
    u_mapping = {old_u: new_u for new_u, old_u in enumerate(unique_users)}
    i_mapping = {old_i: new_i for new_i, old_i in enumerate(unique_items)}
    
    for interaction in interactions:
        interaction['u'] = u_mapping[interaction['u']]
        interaction['i'] = i_mapping[interaction['i']]
        
    # Group by user again
    user_to_items = defaultdict(list)
    for interaction in interactions:
        user_to_items[interaction['u']].append((interaction['pos'], interaction['i']))
        
    for u in user_to_items:
        # Sort by pos to respect original order
        user_to_items[u].sort(key=lambda x: x[0])
        user_to_items[u] = [x[1] for x in user_to_items[u]]
        
    num_users = len(unique_users)
    num_items = len(unique_items)
    num_train_interactions = 0
    
    # 4. Negative sampling for test
    all_items = list(range(num_items))
    
    train_lines = []
    test_ratings = []
    test_negatives = []
    
    for u in range(num_users):
        items = user_to_items[u]
        if len(items) < 2:
            continue
            
        # 3. Leave-one-out split
        test_item = items[-1]
        train_items = items[:-1]
        
        # Train lines
        for i in train_items:
            train_lines.append(f"{u}\t{i}\t1\t0\n")
            num_train_interactions += 1
            
        # Test rating
        test_ratings.append(f"{u}\t{test_item}\t1\t0\n")
        
        # Test negatives
        user_interacted_set = set(items)
        # Sample 99 negatives from NEVER interacted items
        valid_negatives = [i for i in all_items if i not in user_interacted_set]
        neg_samples = random.sample(valid_negatives, 99)
        
        neg_str = "\t".join(str(neg) for neg in neg_samples)
        test_negatives.append(f"({u},{test_item})\t{neg_str}\n")
        
    # 5. Write output files
    train_file = os.path.join(data_dir, 'train.rating')
    test_rating_file = os.path.join(data_dir, 'test.rating')
    test_negative_file = os.path.join(data_dir, 'test.negative')
    
    with open(train_file, 'w') as f:
        f.writelines(train_lines)
        
    with open(test_rating_file, 'w') as f:
        f.writelines(test_ratings)
        
    with open(test_negative_file, 'w') as f:
        f.writelines(test_negatives)
        
    # Matrix sparsity calculated considering entire dense matrix space
    sparsity = 1.0 - (num_train_interactions / (num_users * num_items))
    
    # 6. Print final dataset statistics
    print("-" * 40)
    print("CiteULike-a Preprocessing Complete!")
    print(f"Num Users: {num_users}")
    print(f"Num Items: {num_items}")
    print(f"Num Train Interactions: {num_train_interactions}")
    print(f"Sparsity: {sparsity*100:.4f}%")
    print("-" * 40)

if __name__ == "__main__":
    # Default data dir assuming the script is run from project root, or natively resolved
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'citeulike')
    data_dir = os.path.abspath(data_dir)
    preprocess_citeulike(data_dir)
