import json
import os
import random
import math
import numpy as np
from collections import defaultdict

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def group_by_conference(papers):
    conferences = defaultdict(list)
    for paper in papers:
        conference = paper.get('venue')
        if conference:
            conferences[conference].append(paper)
    return conferences

# def stratified_sampling(conference_papers, target_samples=60):
#     conference_papers.sort(key=lambda x: x['year'])
#     years = [paper['year'] for paper in conference_papers]
#     unique_years = sorted(set(years))
#     print(f"Unique years: {unique_years}")
    
#     # Calculate the number of papers for each year
#     year_counts = {year: years.count(year) for year in unique_years}
#     total_papers = len(conference_papers)
    
#     # Calculate the number of samples per year proportionally
#     sampled_papers = []
#     for year in unique_years:
#         year_papers = [paper for paper in conference_papers if paper['year'] == year]
#         proportion = year_counts[year] / total_papers
#         samples_for_year = max(1, int(target_samples * proportion))
#         print(f"Year {year}: {len(year_papers)} papers, sampling {samples_for_year} papers")
#         if len(year_papers) > samples_for_year:
#             sampled_papers.extend(random.sample(year_papers, samples_for_year))
#         else:
#             sampled_papers.extend(year_papers)

#     # Adjust the total number of sampled papers to be around the target number
#     if len(sampled_papers) > target_samples:
#         sampled_papers = random.sample(sampled_papers, target_samples)
    
#     return sampled_papers

def stratified_sampling(conference_papers, target_samples=60):
    conference_papers.sort(key=lambda x: x['year'])
    years = [paper['year'] for paper in conference_papers]
    unique_years = sorted(set(years))
    print(f"Unique years: {unique_years}")
    
    # Determine the number of strata (at most 10)
    num_years = len(unique_years)
    num_strata = min(10, num_years)
    print(f"Num years: {num_years}, Num strata: {num_strata}")
    strata_size = math.ceil(num_years / num_strata)
    print(f"Strata size: {strata_size}")
    
    # Group years into strata
    strata = []
    for i in range(0, num_years, strata_size):
        strata.append(unique_years[i:i + strata_size])
    print(f"Strata: {len(strata)}")

    # Calculate the number of papers for each strata
    sampled_papers = []
    total_papers = len(conference_papers)
    
    for strat in strata:
        strat_papers = [paper for paper in conference_papers if paper['year'] in strat]
        proportion = len(strat_papers) / total_papers
        # samples_for_strat = max(1, int(target_samples * proportion))
        samples_for_strat = max(1, math.ceil(target_samples * proportion))
        print(f"Strat {strat}: {len(strat_papers)} papers, sampling {samples_for_strat} papers")
        if len(strat_papers) > samples_for_strat:
            sampled_papers.extend(random.sample(strat_papers, samples_for_strat))
        else:
            sampled_papers.extend(strat_papers)

    # Adjust the total number of sampled papers to be around the target number
    if len(sampled_papers) > target_samples:
        sampled_papers = random.sample(sampled_papers, target_samples)
    
    return sampled_papers


def bucket_sample_conferences(conferences, num_conferences=17, papers_per_conference=60):
    sampled_papers = []
    for conference, papers in conferences.items():
        print(f"Conference: {conference}, Papers: {len(papers)}")   
        if len(papers) < papers_per_conference:
            continue
        sampled_papers.extend(stratified_sampling(papers))
        if len(sampled_papers) >= num_conferences * papers_per_conference:
            break
    return sampled_papers[:num_conferences * papers_per_conference]

def save_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    input_file_path = "filtered_data_with_meta_info.jsonl"
    output_file_path = "test_set.jsonl"
    
    papers = load_jsonl(input_file_path)
    conferences = group_by_conference(papers)
    print(f"Total conferences: {len(conferences)}")

    sampled_papers = bucket_sample_conferences(conferences)
    
    save_to_jsonl(sampled_papers, output_file_path)
    
    print(f"Total sampled papers: {len(sampled_papers)}")
    print(sampled_papers[:2])  # Print first two samples for verification
