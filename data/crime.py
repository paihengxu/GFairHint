import pandas as pd
import numpy as np
import json
import os
import us
from county_adjacency import get_neighboring_areas, CountyNotFoundError, supported_areas
import random
from sklearn.model_selection import train_test_split

random.seed(3)


def read_communities_data():
    with open('communities.col_names', 'r') as inf:
        col_names = [line.strip() for line in inf]

    df = pd.read_csv('communities.data', names=col_names)
    return df


def join_datasets():
    comm_df = read_communities_data()
    comm_df.dropna(subset=['ViolentCrimesPerPop'], inplace=True)
    # add columns
    comm_df['avg_crime_score'] = pd.Series(dtype=np.float32)
    comm_df['avg_all_score'] = pd.Series(dtype=np.float32)
    comm_df['crime_score_dist'] = [[0, 0, 0, 0, 0]] * len(comm_df)
    comm_df['crime_score_dist'] = comm_df['crime_score_dist'].astype('object')
    comm_df['county_name'] = pd.Series(dtype=str)

    abbr_fips_mapping = us.states.mapping('abbr', 'fips')
    # We cannot do str.lower(), because the key for finding neighborhood is case sensitive
    areas = supported_areas()
    with open('scraped_niche_scores.jsonl', 'r') as inf:
        last_state = ''
        for line in inf:
            data = json.loads(line.strip())

            # no scraped review info
            if 'avg_crime_score' not in data or data['avg_crime_score'] == -1:
                data['avg_crime_score'] = np.nan
                data['score_dist'] = [0] * 5
                data['all_score'] = np.nan
            tmp_score_dist = [[ele] for ele in data['score_dist']]

            # get state name
            scraped_place_split_list = data['place'].split('-')
            state = scraped_place_split_list[-1].upper()

            # filter by state, as there are cities with same name across states
            if state != last_state:
                try:
                    state_df = comm_df[comm_df['state'] == int(abbr_fips_mapping[state])]
                except KeyError:
                    print(data['place'])
                    print(state)
                    print(state_df)
                    continue


            # join with comm_df
            regex_pattern = ''
            filter_df = state_df
            matched = False
            for ele in scraped_place_split_list:
                regex_pattern += ele
                filter_df = filter_df[filter_df['communityname'].str.contains(f"^{regex_pattern}.*?",
                                                                              case=False, regex=True)]
                if len(filter_df) == 1:
                    idx = filter_df.index.values[0]
                    comm_df.loc[idx, 'avg_crime_score'] = data['avg_crime_score']
                    comm_df.loc[idx, 'crime_score_dist'] = tmp_score_dist
                    comm_df.loc[idx, 'avg_all_score'] = data['all_score']
                    matched = True
                    break
                elif len(filter_df) == 0:
                    break

            # deal with special cases
            if not matched:
                tmp_df = state_df[state_df['communityname'].str.contains(f"^{scraped_place_split_list[0]}(city|town)$",
                                                                         case=False, regex=True)]
                if len(tmp_df) == 1:
                    idx = tmp_df.index.values[0]
                    comm_df.loc[idx, 'avg_crime_score'] = data['avg_crime_score']
                    comm_df.loc[idx, 'crime_score_dist'] = tmp_score_dist
                    comm_df.loc[idx, 'avg_all_score'] = data['all_score']
                else:
                    print(f"No matching for {data['place']}")
                    assert len(tmp_df) == 0, tmp_df[['communityname', 'ViolentCrimesPerPop']]

            # get county adjacency
            county_name = f'County, {state}'
            county_name_list = []
            n = len(scraped_place_split_list)
            found_county = False
            for i in range(2, n):
                county_name = scraped_place_split_list[n - i].capitalize() + ' ' + county_name
                county_name_list.append(county_name)
                if 'Dekalb County' in county_name:
                    county_name = county_name.replace('Dekalb', 'DeKalb')
                elif county_name in ['Fairbanks North Star County, AK']:
                    county_name = county_name.replace('County', 'Borough')
                elif county_name == 'Anchorage County, AK':
                    county_name = 'Anchorage Municipality, AK'
                elif county_name == 'Juneau County, AK':
                    county_name = 'Juneau City and Borough, AK'
                elif 'Miami Dade County, FL' in county_name:
                    county_name = 'Miami-Dade County, FL'
                elif 'Laporte County, IN' in county_name:
                    county_name = 'LaPorte County, IN'
                elif 'Mccracken County, KY' in county_name:
                    county_name = 'McCracken County, KY'
                elif county_name in ['Orleans County, LA', 'Jefferson County, LA', 'Caddo County, LA',
                                     'East Baton Rouge County, LA', 'Calcasieu County, LA', 'Rapides County, LA',
                                     'Iberia County, LA', 'Lafayette County, LA', 'Natchitoches County, LA',
                                     'Acadia County, LA', 'Washington County, LA', 'Bossier County, LA',
                                     'Lincoln County, LA', 'Terrebonne County, LA', 'Jefferson County, LA',
                                     'Ouachita County, LA']:
                    county_name = county_name.replace('County', 'Parish')
                elif county_name == 'Mckinley County, NM':
                    county_name = 'McKinley County, NM'
                elif county_name == 'Mclennan County, TX':
                    county_name = 'McLennan County, TX'
                elif county_name == 'Fond Du Lac County, WI':
                    county_name = 'Fond du Lac County, WI'
                elif '(city)' in county_name:
                    county_name = county_name.replace('(city) County', 'city')

                # print(county_name)
                if county_name in areas:
                    found_county = True
                    comm_df.loc[idx, 'county_name'] = county_name
                    # get_neighboring_areas(county_name)
                    break
            if not found_county:
                print(data['place'])
                print(county_name_list)

    comm_df.to_csv('crime_wtih_adjacent_county.csv')


class CrimeDataset(object):
    """
    Same usage for ogb datasets
    """

    def __init__(self, data_dir='.'):
        self.network_fn = os.path.join(data_dir, 'crime_network.json')
        self.label_fn = os.path.join(data_dir, 'crime_label.json')
        self.data_dir = data_dir
        if os.path.exists(self.network_fn) and os.path.exists(self.label_fn):
            print("loading dataset")
            self.graph, self.labels = self.load_dataset()
            self.index_list = list(range(len(self.labels)))
            self.report_stats()
            return

        print("creating dataset")
        df = pd.read_csv('crime_wtih_adjacent_county.csv')
        # preprocess df
        # df.dropna(subset=['avg_crime_score'], inplace=True)
        # df.reset_index(drop=True, inplace=True)
        print(f'Number of places with scraped avg_crime_score: {len(df.dropna(subset=["avg_crime_score"]))}')

        df = df.round({'avg_crime_score': 0})

        # get node feature columns
        with open('communities.col_names', 'r') as inf:
            feat_cols = [line.strip() for line in inf]
        for col in ['state', 'county', 'community', 'communityname', 'fold', 'ViolentCrimesPerPop']:
            feat_cols.remove(col)

        edge_index = []
        fair_edge_index = []
        linked_fair_nodes = set()
        node_feat = []
        self.labels = []
        self.index_list = []

        # node fair level
        node_fair_level_idx_list = {}
        for level in range(6):
            node_fair_level_idx_list[level] = df[df['avg_crime_score'] == level].index

        df['isViolent'] = df.apply(lambda x: self.get_isviolent(x), axis=1)
        # df['round_avg_crim_score'] = df.apply(lambda x: self.get_round_avg_score(x), axis=1)
        df[feat_cols] = df[feat_cols].replace('?', '-1')
        for index, row in df.iterrows():
            # node features
            try:
                node_feat.append(row[feat_cols].to_numpy(dtype=np.float32))
            except ValueError:
                print("? in data\n", row[feat_cols])
                continue

            # labels
            self.labels.append(row['isViolent'])
            self.index_list.append(index)

            # add edge according to geography adjacency
            if not row['county_name'] is np.nan:
                adj_counties = list(get_neighboring_areas(row['county_name']))
                adj_counties.append(row['county_name'])
                for county in adj_counties:
                    for node_idx in df[df['county_name'] == county].index:
                        if node_idx == index:
                            continue
                        elif [index, node_idx] in edge_index or [node_idx, index] in edge_index:
                            continue
                        edge_index.append([index, node_idx])

            # add fairness graph edge
            if row['avg_crime_score'] is np.nan:
                continue
            if row['avg_crime_score'] not in node_fair_level_idx_list.keys():
                continue
            # same_level_nodes = df[df['avg_crime_score'] == row['avg_crime_score']].index
            # for node_idx in same_level_nodes:
            for node_idx in node_fair_level_idx_list[row['avg_crime_score']]:
                if node_idx in linked_fair_nodes:
                    continue
                if node_idx == index:
                    continue
                elif [index, node_idx] in fair_edge_index or [node_idx, index] in fair_edge_index:
                    continue
                fair_edge_index.append([index, node_idx])
            linked_fair_nodes.add(index)

            if index % 100 == 0:
                print(f"processed {index} nodes")

        self.graph = {
            'edge_index': np.array(edge_index).T,
            'edge_feat': None,
            'node_feat': np.array(node_feat),
            'num_nodes': len(self.labels),
            'fair_edge_index': np.array(fair_edge_index).T
        }
        assert np.array(edge_index).T.shape[0] == 2, np.array(edge_index).shape
        assert np.array(node_feat).shape == (len(self.labels), len(feat_cols)), np.array(node_feat).shape

        # one-off script to generate train/dev/test split
        # tmp_train, test = train_test_split(self.index_list, test_size=0.2)
        # train, dev = train_test_split(tmp_train, test_size=0.25)
        # print([len(ele) for ele in [train, dev, test]])
        # with open('crime_split.json', 'w') as outfile:
        #     outfile.write(f'{json.dumps({"train": train, "valid": dev, "test": test})}')

        self.report_stats()
        self.save_dataset()

    def get_idx_split(self, rand=False):
        if rand:
            with open(os.path.join(self.data_dir, "crime_split.json"), 'r') as inf:
                line = inf.readline()
                split_dict = json.loads(line.strip())
        else:
            tmp_train, test = train_test_split(self.index_list, test_size=0.2)
            train, dev = train_test_split(tmp_train, test_size=0.25)
            print([len(ele) for ele in [train, dev, test]])
            # with open('crime_split.json', 'w') as outfile:
            #     outfile.write(f'{json.dumps({"train": train, "valid": dev, "test": test})}')
            split_dict = {"train": train, "valid": dev, "test": test}
        return split_dict

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.labels

    @staticmethod
    def get_isviolent(row):
        return 1 if row['ViolentCrimesPerPop'] >= 0.15 else 0

    def load_dataset(self):
        with open(self.network_fn, 'r') as inf:
            line = inf.readline()
            graph = json.loads(line.strip())
            for k in ['edge_index', 'node_feat', 'fair_edge_index']:
                graph[k] = np.array(graph[k])

        with open(self.label_fn, 'r') as inf:
            line = inf.readline()
            labels = json.loads(line.strip())

        return graph, labels['labels']

    def save_dataset(self):
        with open(self.network_fn, 'w') as outf:
            graph = self.graph
            for k in ['edge_index', 'node_feat', 'fair_edge_index']:
                graph[k] = graph[k].tolist()
            outf.write(f"{json.dumps(graph)}")

        with open(self.label_fn, 'w') as outf:
            outf.write(f"{json.dumps({'labels': self.labels})}")

    def report_stats(self):
        print(self.graph.keys())
        print('edge_index shape:', self.graph['edge_index'].shape)
        print('node_feat shape:', self.graph['node_feat'].shape)
        print('fair_edge_index shape:', self.graph['fair_edge_index'].shape)
        print('# of labels', len(self.labels))


if __name__ == '__main__':
    join_datasets()
    # test_data = CrimeDataset()
    # graph, label = test_data[0]
    # test_data.get_idx_split()
