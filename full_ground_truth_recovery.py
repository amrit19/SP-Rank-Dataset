######################################################################################## IMPORTS ########################################################################################

import pandas as pd
import numpy as np
import random
import ast
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm
from scipy.interpolate import make_interp_spline

######################################################################################## Voting Rules ########################################################################################

# Function to apply Borda Voting method
def borda_voting_rule(votes):
    scores = defaultdict(int)
    num_options = len(max(votes, key=len))  # get the number of options based on the longest vote

    for vote in votes:
        for idx, option in enumerate(vote):
            scores[option] += num_options - idx  # assign points based on the position in the vote

    # Return only the options, not the scores, sorted in descending order of scores
    return [option for option, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

# Function to apply Copeland Voting method
def copeland_voting_rule(votes):
    # Count the frequency of each alternative
    frequency = defaultdict(int)
    for vote_list in votes:
        vote = vote_list[0] 
        frequency[vote] += 1
    scores = {alternative: 0 for alternative in frequency}
    for alt_a in frequency:
        for alt_b in frequency:
            if alt_a == alt_b:
                continue
            if frequency[alt_a] > frequency[alt_b]:
                scores[alt_a] += 1
            elif frequency[alt_a] < frequency[alt_b]:
                scores[alt_a] -= 1

    # Sort alternatives based on the Copeland scores
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [key for key, value in sorted_scores]

# Function to apply Maximin method
def maximin_voting_rule(votes):
    scores = defaultdict(lambda: float('-inf'))  # Initialize scores with negative infinity
    
    # Iterate through each vote
    for vote in votes:
        for idx, option in enumerate(vote):
            # Update the score of an option to its highest minimum position
            scores[option] = max(scores[option], len(vote) - idx)
    
    # Sort options based on their highest minimum score in descending order
    sorted_options = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return only the options, sorted by their highest minimum score
    return [option for option, score in sorted_options]

# Function to apply Schulze Voting method
def schulze_voting_rule(votes):
    # Create set of all alternatives
    alternatives_set = set()
    for vote in votes:
        alternatives_set.update(vote)

    # Create mapping from alternatives to indices
    alternatives = list(alternatives_set)
    alternatives_to_indices = {alternative: i for i, alternative in enumerate(alternatives)}

    # Initialize pairwise preference matrix and strongest paths matrix
    num_alternatives = len(alternatives)
    pairwise_pref = np.zeros((num_alternatives, num_alternatives))
    strongest_paths = np.zeros((num_alternatives, num_alternatives))

    # Compute pairwise preference matrix
    for vote in votes:
        for i, j in combinations(vote, 2):
            if vote.index(i) < vote.index(j):
                pairwise_pref[alternatives_to_indices[i]][alternatives_to_indices[j]] += 1
            else:
                pairwise_pref[alternatives_to_indices[j]][alternatives_to_indices[i]] += 1

    # Initialize strongest paths matrix with pairwise preferences
    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                strongest_paths[i][j] = pairwise_pref[i][j]

    # Compute strongest paths matrix
    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                for k in range(num_alternatives):
                    if k != i and k != j:
                        strongest_paths[j][k] = max(strongest_paths[j][k], min(strongest_paths[j][i], strongest_paths[i][k]))

    # Compute ranking
    ranking = []
    for i in range(num_alternatives):
        rank = sum(1 for j in range(num_alternatives) if strongest_paths[i][j] > strongest_paths[j][i])
        ranking.append((alternatives[i], rank))

    # Sort alternatives by rank in descending order
    ranking.sort(key=lambda x: -x[1])

    return [alternative for alternative, rank in ranking]

################################################################# SP Format Adapted Vote Aggregation #########################################

def SP_borda_rule(votes):
    unique_alts = set(alt for ranking in votes for alt in ranking)
    borda_scores = {alt: 0 for alt in unique_alts}
    
    for ranking in votes:
        num_options = len(ranking)
        for i, alt in enumerate(ranking):
            borda_scores[alt] += num_options - i - 1
    
    sorted_alts = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
    return [option for option, score in sorted_alts]

def SP_copeland_rule(votes):

    # Create a dictionary to store pairwise comparisons
    pairwise_comparisons = {}

    # Fill the pairwise comparison dictionary with counts of wins for each pair
    for ranking in votes:
        for i, winner in enumerate(ranking):
            for loser in ranking[i+1:]:
                if (winner, loser) not in pairwise_comparisons:
                    pairwise_comparisons[(winner, loser)] = 0
                pairwise_comparisons[(winner, loser)] += 1

    # Calculate the Copeland scores
    unique_alts = set(alt for ranking in votes for alt in ranking)
    copeland_scores = defaultdict(int)

    for alt in unique_alts:
        wins = sum(1 for (winner, loser) in pairwise_comparisons if winner == alt)
        losses = sum(1 for (winner, loser) in pairwise_comparisons if loser == alt)
        copeland_scores[alt] = wins - losses

    # Sort the alternatives based on their Copeland scores
    sorted_alts = sorted(copeland_scores.items(), key=lambda x: x[1], reverse=True)

    # Return only the options, not the copelands
    return [option for option, count in sorted_alts]

def SP_maximin_rule(rankings):
    scores = defaultdict(int)
    unique_alts = set(alt for ranking in rankings for alt in ranking)
    maximin_scores = {alt: len(unique_alts) for alt in unique_alts}

    for ranking in rankings:
        for i, alt in enumerate(ranking):
            if scores[alt] > i or scores[alt] == 0:
                scores[alt] = i  # assign the minimum position the option has ever appeared
        for alt, score in scores.items():
            if score < maximin_scores[alt]:
                maximin_scores[alt] = score

    # Sort the alternatives based on their Maximin scores
    sorted_alts = sorted(maximin_scores.items(), key=lambda x: x[1])
    # Return only the options, not the maximins
    return [option for option, count in sorted_alts]

def SP_schulze_rule(rankings):
    # Create a set of unique alternatives
    alternatives = set(alt for options in rankings for alt in options)

    # Compute the pairwise preferences
    pairwise_preferences = defaultdict(int)
    for ranking in rankings:
        for i, winner in enumerate(ranking):
            for loser in ranking[i + 1:]:
                if winner in alternatives and loser in alternatives:
                    pairwise_preferences[(winner, loser)] += 1

    # Compute the strongest path matrix using Floyd-Warshall algorithm
    strongest_paths = {a: {b: 0 for b in alternatives} for a in alternatives}

    for i in alternatives:
        for j in alternatives:
            if i != j:
                if pairwise_preferences[(i, j)] > pairwise_preferences[(j, i)]:
                    strongest_paths[i][j] = pairwise_preferences[(i, j)]

    for i in alternatives:
        for j in alternatives:
            if i != j:
                for k in alternatives:
                    if i != k and j != k:
                        strongest_paths[j][k] = max(strongest_paths[j][k], min(strongest_paths[j][i], strongest_paths[i][k]))

    # Rank the alternatives based on the strongest path matrix
    sorted_alternatives = sorted(alternatives, key=lambda x: sum(strongest_paths[x][a] > strongest_paths[a][x] for a in alternatives), reverse=True)

    # Return only the options, not the schulzs
    return sorted_alternatives


################################################################## Vote Function ########################################################################################

#This is where we define the voting rules we want to use
def votes_function(votes, aggregation_type, rule):
        
    if rule == 'Borda':
        if aggregation_type == 'vote':
            new_alts=borda_voting_rule(votes)
            return new_alts
        if aggregation_type == 'sp':
            new_alts=SP_borda_rule(votes)
            return new_alts
        
    if rule == 'Copeland':
        if aggregation_type == 'vote':
            new_alts=copeland_voting_rule(votes)
            return new_alts
        if aggregation_type == 'sp':
            new_alts=SP_copeland_rule(votes)
            return new_alts

    if rule == 'Maximin':
        if aggregation_type == 'vote':
            new_alts=maximin_voting_rule(votes)
            return new_alts
        if aggregation_type == 'sp':
            new_alts=SP_maximin_rule(votes)
            return new_alts
        
    if rule == 'Schulze':
        if aggregation_type == 'vote':
            new_alts=schulze_voting_rule(votes)
            return new_alts
        if aggregation_type == 'sp':
            new_alts=SP_schulze_rule(votes)
            return new_alts
################################################################## Partial-SP Algorithm ########################################################################################

#If in all votes, a is preferred over b return 1 else return 0
#We see for a pair of alternatives, how many times a is preferred over b
def infoab(votes, treatment, a, b):

    if treatment == 4 or treatment == 5 or treatment == 6: #Elicitation Formats - Rank-None, Rank-Top, Rank-Rank

        alts = votes
        idxa = alts.index(a)
        idxb = alts.index(b)
        if idxa < idxb:
            return 1
        elif idxa > idxb:
            return 0
        else:
            print('Cant find a or b')

    if treatment == 1 or treatment == 2 or treatment == 3 or treatment == 8: #Elicitation Formats - Top-None, Top-Top, Top-Approval(3), Top-Rank

        alts = votes[0]

        if alts == a:
            return 1
        elif alts == b:
            return 0
        else:
            return -1
        
    if treatment == 9: #Elicitation Formats - Approval(2) - Approval(2)

        alts1 = votes[0]
        alts2 = votes[1]

        if alts1 == a or alts2 == a:
            return 1
        elif alts1 == b or alts2 == b:
            return 0
        else:
            return -1

    if treatment == 7: #Elicitation Formats - Approval(3) - Rank
        alts1 = votes[0]
        alts2 = votes[1]
        alts3 = votes[2]

        if alts1 == a or alts2 == a or alts3 == a:
            return 1
        elif alts1 == b or alts2 == b or alts3 == b:
            return 0
        else:
            return -1

#If in all predictions, a is preferred over b return alpha else return beta
def predab(predictions, treatment, a, b, alpha, beta):

    if treatment == 3 or treatment == 6 or treatment ==7: #Elicitation Formats - Top-Rank, Approval(3) - Rank, Rank-Rank

        pred_alts = predictions
        idxa = pred_alts.index(a)
        idxb = pred_alts.index(b)
        if idxa < idxb:
            return alpha
        elif idxa > idxb:
            return beta
        else:
            print('Same location for prediction report')
    
    if treatment == 2 or treatment == 5: #Elicitation Formats - Top-Top, Rank-Top

        pred_alts = predictions[0]
        if pred_alts == a:
            return alpha
        elif pred_alts == b:
            return beta
        else:
            return 0.5
            
    if treatment == 8: #Elicitation Formats - Top-Approval(3)

        pred_alts1 = predictions[0]
        pred_alts2 = predictions[1]
        pred_alts3 = predictions[2]
        if pred_alts1 == a or pred_alts2 == a or pred_alts3 == a:
            return alpha
        elif pred_alts1 == b or pred_alts2 == b or pred_alts3 == b:
            return beta
        else:
            return 0.5
    if treatment == 9: #Elicitation Formats - Approval(2) - Approval(2)

        pred_alts1 = predictions[0]
        pred_alts2 = predictions[1]
        if pred_alts1 == a or pred_alts2 == a:
            return alpha
        elif pred_alts1 == b or pred_alts2 == b:
            return beta
        else:
            return 0.5

#For each pair of alternatives, check how many times a wins over b and how many times a loses over b and also check what is the prediction
#for each case, then calculate the prediction-normalized score for the pair of a and b and then see who wins.
def Aggregate_II(information, prediction):
    idx1 = information[information == 1].index
    idx0 = information[information == 0].index

    
    prediction_0 = pd.Series([1 - x for x in prediction], index=prediction.index)
    p11 = np.mean(prediction[idx1])
    p10 = np.mean(prediction[idx0])
    p01 = np.mean(prediction_0[idx1])
    p00 = np.mean(prediction_0[idx0])
    if len(idx1) == 0 or len(idx0) == 0:
        return 0
    nv1 = len(idx1) / (len(idx1) + len(idx0)) * (1 + p01/p10)
    nv0 = len(idx0) / (len(idx1) + len(idx0)) * (1 + p10/p01)
    
    if nv1 >= nv0:
        return 1
    else:
        return 0

def complete_ranking(lpairs):
    alts = []
    for v in lpairs:
        alts.extend([v[0], v[1]])
    alts = list(set(alts))
    score = [0] * len(alts)
    for v in lpairs:
        pos = alts.index(v[0])
        score[pos] += 1

    # Sort the alternatives by their scores
    sorted_alts = sorted(zip(alts, score), key=lambda x: x[1], reverse=True)
    # Extract the sorted alternatives only
    ranking = [x[0] for x in sorted_alts]

    return ranking


def sp_voting(df, treatment):
    rankings_df = pd.DataFrame(columns=['domain', 'questions', 'ranking'])
    #Dropping the duplicates and converting the options to tuple
    df['options'] = df['options'].apply(lambda x: tuple(x) if not isinstance(x, tuple) else x)

    #Extracting the unique questions
    Questions = df[['domain', 'questions', 'options']].drop_duplicates()

    #Grouping the dataframe by domain
    Q = Questions.groupby('domain').filter(lambda x: len(x) > 0)
    df=df.groupby('domain').filter(lambda x: len(x) > 0)

    #Parameters for the SP voting rule

    alpha_0 = 0.55
    beta_0 = 0.1

    for index_test, row_test in Q.iterrows():
        
        #We are extracting all the data for a particular subset and storing it in dfsub
        dfsub = df.loc[(df['domain'] == Q.loc[index_test]['domain']) & (df['questions'] == Q.loc[index_test]['questions']) & (df['treatment'] == treatment), :].copy()
        options=Q.loc[index_test]['options']
        pairs=list(combinations(options, 2))
        ordered_pairs = []
        for v in pairs:
            v1 = int(v[0])
            v2 = int(v[1])
            dfsub.loc[:, 'information'] = dfsub.apply(lambda x: infoab(x['votes'], x['treatment'], v1, v2), axis=1)
            #dfsub.loc[:, 'information'] = dfsub['votes'].map(lambda x: infoab(x, v1, v2))
            dfsub.loc[:, 'prediction'] = dfsub.apply(lambda x: predab(x['predictions'], x['treatment'], v1, v2, alpha_0, beta_0), axis=1)
            agg_alt = Aggregate_II(dfsub['information'], dfsub['prediction'])
            if agg_alt == 1:
                ordered_pairs.append([v1, v2])
            else:
                ordered_pairs.append([v2, v1])
        ranking = complete_ranking(ordered_pairs)
        # Check if the row already exists in the DataFrame
        row_exists = rankings_df.loc[(rankings_df['domain'] == row_test['domain']) & (rankings_df['questions'] == row_test['questions'])].shape[0] > 0
        if not row_exists:
            new_rows = []
            new_rows.append(pd.DataFrame({'domain': [row_test['domain']], 'questions': [row_test['questions']], 'ranking': [ranking]}))
            rankings_df = pd.concat([rankings_df] + new_rows, ignore_index=True)
        #This is where the voting rule comes which we will use to aggregate the sp votes
    final_ranking = rankings_df['ranking']
    return final_ranking

##################################################################  Partial-SP ########################################################################################


def run_partial_sp(elicitation_format, rule, domain):
    # Find the corresponding key in the dictionary
    treatment = None
    for key, value in map_elicitation.items():
        if value == elicitation_format:
            treatment = key
    if elicitation_format == 'approval3-rank':
        elicitation_format = 'subset3-rank'
    if elicitation_format == 'top-approval3':
        elicitation_format = 'top-subset3'
    if elicitation_format == 'approval2-approval2':
        elicitation_format = 'subset2-subset2' 

    # Read CSV file
    df = pd.read_csv(f'Elicitation Formats/{elicitation_format}/{elicitation_format}_{domain}_4.csv')
    # Convert the string representation of lists in 'votes' to actual list
    df['votes'] = df['votes'].apply(ast.literal_eval)
    df['predictions'] = df['predictions'].apply(ast.literal_eval)
    df['options'] = df['options'].apply(ast.literal_eval)

    # Lists to store the Kendall Tau distances and Spearman correlations
    kendall_tau_sp = []
    kendall_tau_vote = []
    spearman_rho_sp = []
    spearman_rho_vote = []

    # Number of bootstrap samples
    num_bootstraps = 1000

    # Perform bootstrapping
    for control in tqdm(range(num_bootstraps), desc="Bootstrapping", unit="bootstrap"):
        # Group the data by question
        grouped = df.groupby('questions')

        bootstrap_sample_dfs = []  # List to store each group's sampled DataFrame
        sp_partial_ground_truth = pd.DataFrame()
        for name, group in grouped:
            sampled_df = group.sample(20, replace=True)
            bootstrap_sample_dfs.append(sampled_df)

        # Concatenate all sampled DataFrames
        bootstrap_sample_df = pd.concat(bootstrap_sample_dfs, ignore_index=True)

        sp_partial_ground_truth = sp_voting(bootstrap_sample_df, treatment)

        #At this point we have partial ground truths for each subset

        sp_ranking=votes_function(sp_partial_ground_truth, aggregation_type='sp', rule=rule)
        vote_ranking = votes_function(bootstrap_sample_df['votes'], aggregation_type='vote', rule=rule)
        ground_truth_ranking = sorted (vote_ranking)

        common_alternatives = set(sp_ranking) & set(vote_ranking)

        ground_truth_ranking = [alt for alt in ground_truth_ranking if alt in common_alternatives]
        vote_ranking = [alt for alt in vote_ranking if alt in common_alternatives]
        sp_ranking = [alt for alt in sp_ranking if alt in common_alternatives]

        # Calculate Kendall Tau distance between ground_truth and sp_ranking
        tau_sp, _ = kendalltau(ground_truth_ranking, sp_ranking)
        kendall_tau_sp.append(tau_sp)

        # Calculate Kendall Tau distance between ground_truth and vote_ranking
        tau_vote, _ = kendalltau(ground_truth_ranking, vote_ranking)
        kendall_tau_vote.append(tau_vote)

        # Calculate Spearman correlation between ground_truth and sp_ranking
        rho_sp, _ = spearmanr(ground_truth_ranking, sp_ranking)
        spearman_rho_sp.append(rho_sp)

        # Calculate Spearman correlation between ground_truth and vote_ranking
        rho_vote, _ = spearmanr(ground_truth_ranking, vote_ranking)
        spearman_rho_vote.append(rho_vote)    

    ######################################################################################## Plotting ########################################################################################

    kendall_tau_sp = pd.DataFrame(kendall_tau_sp, columns=['Kendall_Tau_SP'])
    kendall_tau_vote = pd.DataFrame(kendall_tau_vote, columns=['Kendall_Tau_Vote'])
    spearman_rho_sp = pd.DataFrame(spearman_rho_sp, columns=['Spearman_Rho_SP'])
    spearman_rho_vote = pd.DataFrame(spearman_rho_vote, columns=['Spearman_Rho_Vote'])
    
    mean_tau_sp = np.mean(kendall_tau_sp['Kendall_Tau_SP'])
    ci_tau_sp = np.percentile(kendall_tau_sp['Kendall_Tau_SP'], [2.5, 97.5])
    mean_tau_vote = np.mean(kendall_tau_vote['Kendall_Tau_Vote'])
    ci_tau_vote = np.percentile(kendall_tau_vote['Kendall_Tau_Vote'], [2.5, 97.5])

    mean_rho_sp = np.mean(spearman_rho_sp['Spearman_Rho_SP'])
    ci_rho_sp = np.percentile(spearman_rho_sp['Spearman_Rho_SP'], [2.5, 97.5])
    mean_rho_vote = np.mean(spearman_rho_vote['Spearman_Rho_Vote'])
    ci_rho_vote = np.percentile(spearman_rho_vote['Spearman_Rho_Vote'], [2.5, 97.5])

    # Data preparation for plotting
    metrics = ['Kendall Tau SP', 'Kendall Tau Vote', 'Spearman Rho SP', 'Spearman Rho Vote']
    means = [mean_tau_sp, mean_tau_vote, mean_rho_sp, mean_rho_vote]
    ci_lower_bounds = [ci_tau_sp[0], ci_tau_vote[0], ci_rho_sp[0], ci_rho_vote[0]]
    ci_upper_bounds = [ci_tau_sp[1], ci_tau_vote[1], ci_rho_sp[1], ci_rho_vote[1]]
    errors = [(np.array(means) - np.array(ci_lower_bounds)), (np.array(ci_upper_bounds) - np.array(means))]

    # Plotting the bar plots with error bars
    # plt.figure(figsize=(12, 6))
    # bar_positions = range(len(metrics))
    # plt.bar(bar_positions, means, yerr=errors, align='center', alpha=0.7, capsize=10, color=['red', 'blue', 'green', 'purple'])
    # plt.xticks(bar_positions, metrics)
    # plt.ylabel('Values')
    # plt.title('Mean and 95% Confidence Intervals for Correlation Metrics')
    # plt.grid(True, linestyle='--', alpha=0.6)
    # #plt.show()
    return mean_tau_sp, ci_tau_sp, mean_tau_vote, ci_tau_vote, mean_rho_sp, ci_rho_sp, mean_rho_vote, ci_rho_vote
    

######################################################################################## Main Function ##################################################################

import pandas as pd
import ast
import os
# Load your combined responses CSV
df = pd.read_csv('SP_Rank_Dataset.csv')

# Filter for workerid >= 721
df = df[df['workerid'] <= 720].copy()

# Fix types for relevant columns
for col in ['votes', 'predictions', 'options']:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Normalize elicitation format for directory compatibility
df['elicitation_format'] = df['elicitation_format'].str.lower().str.replace('subset3', 'approval3').str.replace('subset2', 'approval2')

# Define your map
map_elicitation = {
    1: 'top-none', 2: 'top-top', 3: 'top-rank',
    4: 'rank-none', 5: 'rank-top', 6: 'rank-rank',
    7: 'approval3-rank', 8: 'top-approval3', 9: 'approval2-approval2'
}
reverse_map_elicitation = {v: k for k, v in map_elicitation.items()}

# Pick voting rule for comparison
voting_rules = ['Borda', 'Copeland', 'Maximin']


performance_records = []

for elicitation_format in df['elicitation_format'].unique():
    print(f"\n🔍 Running Partial-SP for Elicitation Format: {elicitation_format}")

    if elicitation_format in ['top-none', 'rank-none', 'approval3-rank', 'top-approval3', 'approval2-approval2']:
        print(f"⏭️ Skipping ground truth evaluation for '{elicitation_format}' — lacks complete preference data.")
        continue

    domains = df[df['elicitation_format'] == elicitation_format]['domain'].unique()

    for domain in domains:
        print(f"\n🌍 Processing domain: {domain}")
        subset = df[(df['elicitation_format'] == elicitation_format) & (df['domain'] == domain)].copy()
        filename_fmt = elicitation_format.replace('approval3', 'subset3').replace('approval2', 'subset2')
        csv_path = f"Elicitation Formats/{filename_fmt}/{filename_fmt}_{domain}_4.csv"

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        subset.to_csv(csv_path, index=False)

        for voting_rule in voting_rules:
            print(f"🧪 Evaluating rule: {voting_rule}")
            mean_tau_sp, ci_tau_sp, mean_tau_vote, ci_tau_vote, \
            mean_rho_sp, ci_rho_sp, mean_rho_vote, ci_rho_vote = run_partial_sp(elicitation_format, voting_rule, domain)

            performance_records.append({
                'Elicitation Format': elicitation_format,
                'Domain': domain,
                'Voting Rule': voting_rule,
                'Kendall Tau SP (Mean)': mean_tau_sp,
                'Kendall Tau SP (CI Lower)': ci_tau_sp[0],
                'Kendall Tau SP (CI Upper)': ci_tau_sp[1],
                'Kendall Tau Vote (Mean)': mean_tau_vote,
                'Kendall Tau Vote (CI Lower)': ci_tau_vote[0],
                'Kendall Tau Vote (CI Upper)': ci_tau_vote[1],
                'Spearman Rho SP (Mean)': mean_rho_sp,
                'Spearman Rho SP (CI Lower)': ci_rho_sp[0],
                'Spearman Rho SP (CI Upper)': ci_rho_sp[1],
                'Spearman Rho Vote (Mean)': mean_rho_vote,
                'Spearman Rho Vote (CI Lower)': ci_rho_vote[0],
                'Spearman Rho Vote (CI Upper)': ci_rho_vote[1]
            })
# Save performance comparison table
performance_df = pd.DataFrame(performance_records)
performance_df.to_csv("voting_rule_performance_comparison_4.csv", index=False)

print("\n📊 Voting Rule Performance Table (Per-Domain):")
print(performance_df)

# Optional: Display average performance across all domains
avg_performance = performance_df.groupby(['Elicitation Format', 'Voting Rule']).mean(numeric_only=True).reset_index()
avg_performance.to_csv("voting_rule_average_performance_4.csv", index=False)
print("\n📈 Average Performance Across Domains:")
print(avg_performance)
