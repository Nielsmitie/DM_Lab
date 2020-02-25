# Register of all commands to generate the images for the presentation

## Agnos s

### Paper comparison

```shell script
--path results/paper_agnos_s_reworked.csv --model agnos_s --df_name paper_agnos_s --save pics --title "Comparing our results with the original paper: "
```

### Comparing our model against competitor
```shell script
`--path results/paper_agnos_s_reworked.csv --second_path results/spec-1.csv --other_df_name paper_spec --df_name paper_agnos_s --save pics --title "Comparing Agnos S with SPEC: "`
```

### Comparing behaviour with train/test split

```shell script
--path results/train_test_paper_agnos_s_reworked.csv --second_path results/spec_train_test_reworked.csv --model "agnos_s" --other_df_name SPEC --df_name "Agnos S" --save pics --title "Comparing Agnos S behaviour with SPEC on unseen data: "
```