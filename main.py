"""
**Author:** Brayden Edwards  
**Course:** CS 541 – Machine Learning Challenges in the Real World
**School:** Oregon State University
**Professor:** Dr. Kiri Wagstaff  
**Term:** Spring 2025  
"""
import expert_model
import home_model
import representation
import explanation


def main():
    # Challenge 1
    print("\n\nRunning class imbalance testing\n")
    expert_model.class_imbalance_test()
    home_model.class_imbalance_test()

    # Challenge 2
    print("\n\nRunning LIME explanation generation\n")
    explanation.get_lime_explanation()

    print("\n\nRunning SHAP explanation generation (local)\n")
    explanation.get_shap_importance()

    print("\n\nRunning Permutation Importance analysis\n")
    explanation.get_permutation_importance()

    print("\n\nRunning SHAP explanation generation (global)\n")
    explanation.get_shap_importance(is_global=True)

    print("\n\nGenerating counterfactual explanations\n")
    explanation.get_counterfactuals()

    # Challenge 3
    print("\n\nRunning representation testing\n")
    representation.representation_test()


if __name__ == "__main__":
    main()