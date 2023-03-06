from sentence_pattern_from_ner_predict import specPatternAnalysis


if __name__=='__main__':
    file_path = 'data/AZ_Corporate_spec_with_studycase.xlsx'
    model_path = 'model-best-sm/'
    spec_col = 'Map Definition'
    save_path = 'data/df_cluster.xlsx'

    df_cluster = specPatternAnalysis(file_path=file_path, model_path=model_path, spec_col=spec_col, k=5)

    df_cluster.to_excel(save_path, index=False)







