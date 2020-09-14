TRAIN_DIR = "./datasets/images_training_rev1/images/"
TRAIN_CSV = "./classes/training_solutions_rev1.csv"

BATCH_SIZE = 128
DEVICE = "cuda" 
VALIDATION_SPLIT = 0.05
RANDOM_SEED = 42
SHUFFLE_DS = True

CSV_HEADER = ['objid', 'sample', 'asset_id', 'dr7objid',
       't01_smooth_or_features_a01_smooth_fraction',
       't01_smooth_or_features_a02_features_or_disk_fraction',
       't01_smooth_or_features_a03_star_or_artifact_fraction',
       't02_edgeon_a04_yes_fraction', 't02_edgeon_a05_no_fraction',
       't03_bar_a06_bar_fraction', 't03_bar_a07_no_bar_fraction',
       't04_spiral_a08_spiral_fraction', 't04_spiral_a09_no_spiral_fraction',
       't05_bulge_prominence_a10_no_bulge_fraction',
       't05_bulge_prominence_a11_just_noticeable_fraction',
       't05_bulge_prominence_a12_obvious_fraction',
       't05_bulge_prominence_a13_dominant_fraction',
       't06_odd_a14_yes_fraction', 't06_odd_a15_no_fraction',
       't07_rounded_a16_completely_round_fraction',
       't07_rounded_a17_in_between_fraction',
       't07_rounded_a18_cigar_shaped_fraction',
       't08_odd_feature_a19_ring_fraction',
       't08_odd_feature_a20_lens_or_arc_fraction',
       't08_odd_feature_a21_disturbed_fraction',
       't08_odd_feature_a22_irregular_fraction',
       't08_odd_feature_a23_other_fraction',
       't08_odd_feature_a24_merger_fraction',
       't08_odd_feature_a38_dust_lane_fraction',
       't09_bulge_shape_a25_rounded_fraction',
       't09_bulge_shape_a26_boxy_fraction',
       't09_bulge_shape_a27_no_bulge_fraction',
       't10_arms_winding_a28_tight_fraction',
       't10_arms_winding_a29_medium_fraction',
       't10_arms_winding_a30_loose_fraction', 't11_arms_number_a31_1_fraction',
       't11_arms_number_a32_2_fraction', 't11_arms_number_a33_3_fraction',
       't11_arms_number_a34_4_fraction',
       't11_arms_number_a36_more_than_4_fraction',
       't11_arms_number_a37_cant_tell_fraction']
