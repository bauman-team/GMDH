namespace GMDH {

class GmdhModel;

/**
 * @brief Enum class for specifying the QR decomposition method for linear equations solving in models.
 * 
 * Detailed description of methods can be found [here](https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html)
 */
enum class Solver { 
    fast, //!< Fast solution with perhaps not the best accuracy using HouseholderQR decomposition
    accurate, //!< Slow solution with maximum accuracy using FullPivHouseholderQR decomposition
    balanced //!< Balanced solution with average speed and accuracy using ColPivHouseholderQR decomposition
};

/**
 * @brief Enum class for specifying the external criterion \f$ E \f$ to select the optimum solution.
 * 
 * Table of symbols:\n
 * \f$ A \f$ - training data;\n
 * \f$ B \f$ - testing data;\n
 * \f$ C \f$ - learning data that is equal to \f$ A\cup{B} \f$;\n
 * \f$ X_A \f$ - input variables matrix of the training data \f$ A \f$;\n
 * \f$ X_B \f$ - input variables matrix of the testing data \f$ B \f$;\n
 * \f$ X_C \f$ - input variables matrix of the learning data\f$ C \f$ that is equal to \f$ X_A\cup{X_B} \f$ ;\n
 * \f$ y_A \f$ - target values vector of the training data \f$ A \f$;\n
 * \f$ y_B \f$ - target values vector of the testing data \f$ B \f$;\n
 * \f$ y_C \f$ - target values vector of the learning data \f$ C \f$ that is equal to \f$ y_A\cup{y_B} \f$;\n
 * \f$ \hat{w}_A \f$ - coefficients vector of the model trained on a training data \f$ A \f$;\n
 * \f$ \hat{w}_B \f$ - coefficients vector of the model trained on a testing data \f$ B \f$;\n
 * \f$ \hat{w}_C \f$ - coefficients vector of the model trained on a learning data \f$ C \f$;\n
 */
enum class CriterionType {
    regularity, //!< \f$ E=||y_{B}-X_{B}\hat{w}_A||^2 \f$
    symRegularity, //!< \f$ E=||y_{B}-X_{B}\hat{w}_A||^2+||y_{A}-X_{A}\hat{w}_B||^2 \f$
    stability, //!< \f$ E=||y_{C}-X_{C}\hat{w}_A||^2 \f$
    symStability, //!< \f$ E=||y_{C}-X_{C}\hat{w}_A||^2 + ||y_{C}-X_{C}\hat{w}_B||^2 \f$
    unbiasedOutputs, //!< \f$ E=||X_{B}\hat{w}_A-X_{B}\hat{w}_B||^2 \f$
    symUnbiasedOutputs, //!< \f$ E=||X_{C}\hat{w}_A-X_{C}\hat{w}_B||^2 \f$
    unbiasedCoeffs, //!< \f$ E=||\hat{w}_A-\hat{w}_B||^2 \f$
    absoluteNoiseImmunity, //!< \f$ E=(X_B\hat{w}_C-X_B\hat{w}_A)^T(X_B\hat{w}_B-X_B\hat{w}_C) \f$
    symAbsoluteNoiseImmunity //!< \f$ E=(X_C\hat{w}_C-X_C\hat{w}_A)^T(X_C\hat{w}_B-X_C\hat{w}_C) \f$
}; // TODO: maybe add cross validation criterion

/// @brief Structure for storing coefficients and predicted values calculated in different ways
struct BufferValues {
    VectorXd coeffsTrain; //!< Coefficients vector calculated using training data
    VectorXd coeffsTest; //!< Coefficients vector calculated using testing data
    VectorXd coeffsAll; //!< Coefficients vector calculated using learning data
    VectorXd yPredTrainByTrain; //!< Predicted values for *training* data calculated using coefficients vector calculated on *training* data
    VectorXd yPredTrainByTest; //!< Predicted values for *training* data calculated using coefficients vector calculated on *testing* data
    VectorXd yPredTestByTrain; //!< Predicted values for *testing* data calculated using coefficients vector calculated on *training* data
    VectorXd yPredTestByTest; //!< Predicted values for *testing* data calculated using coefficients vector calculated on *testing* data
};

/// @brief Class that implements calculations of internal and individual external criterions
class GMDH_API Criterion {
protected:

    CriterionType criterionType; //!< Selected CriterionType object
    Solver solver; //!< Selected Solver object

    /**
     * @brief Implements the internal criterion calculation
     * 
     * @param xTrain Matrix of input variables that should be used to calculate the model coefficients
     * @param yTrain Target values vector for the corresponding xTrain parameter
     * @return Coefficients vector representing a solution of the linear equations system constructed from the parameters data
     */
    VectorXd findBestCoeffs(const MatrixXd& xTrain, const VectorXd& yTrain) const;

    /**
     * @brief Calculate the value of the selected external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param _criterionType Selected external criterion type
     * @param bufferValues Temporary storage for calculated coefficients and target values
     * @return The value of external criterion and calculated model coefficients
     */
    PairDVXd getResult(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, 
                       CriterionType _criterionType, BufferValues& bufferValues) const;

    /**
     * @brief Calculate the regularity external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param bufferValues Temporary storage for calculated coefficients and target values
     * @param inverseSplit True, if it is necessary to swap the roles of training and testing data, otherwise false
     * @return The value of the regularity external criterion and calculated model coefficients
     */
    PairDVXd regularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, 
                        BufferValues& bufferValues, bool inverseSplit = false) const;

    /**
     * @brief Calculate the symmetric regularity external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param bufferValues Temporary storage for calculated coefficients and target values
     * @return The value of the symmertic regularity external criterion and calculated model coefficients
     */
    PairDVXd symRegularity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest, 
                           BufferValues& bufferValues) const;

    /**
     * @brief Calculate the stability external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param bufferValues Temporary storage for calculated coefficients and target values
     * @param inverseSplit True, if it is necessary to swap the roles of training and testing data, otherwise false
     * @return The value of the stability external criterion and calculated model coefficients
     */
    PairDVXd stability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                       BufferValues& bufferValues, bool inverseSplit = false) const;

    /**
     * @brief Calculate the symmetric stability external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param bufferValues Temporary storage for calculated coefficients and target values
     * @return The value of the symmertic stability external criterion and calculated model coefficients
     */
    PairDVXd symStability(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                          BufferValues& bufferValues) const;

    /**
     * @brief Calculate the unbiased outputs external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param bufferValues Temporary storage for calculated coefficients and target values
     * @return The value of the unbiased outputs external criterion and calculated model coefficients
     */
    PairDVXd unbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                             BufferValues& bufferValues) const;

    /**
     * @brief Calculate the symmetric unbiased outputs external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param bufferValues Temporary storage for calculated coefficients and target values
     * @return The value of the symmetric unbiased outputs external criterion and calculated model coefficients 
     */
    PairDVXd symUnbiasedOutputs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                                BufferValues& bufferValues) const;

    /**
     * @brief Calculate the unbiased coefficients external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param bufferValues Temporary storage for calculated coefficients and target values
     * @return The value of the unbiased coefficients external criterion and calculated model coefficients  
     */
    PairDVXd unbiasedCoeffs(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                            BufferValues& bufferValues) const;

    /**
     * @brief Calculate the absolute noise immunity external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param bufferValues Temporary storage for calculated coefficients and target values 
     * @return The value of the absolute noise immunity external criterion and calculated model coefficients 
     */
    PairDVXd absoluteNoiseImmunity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                               BufferValues& bufferValues) const;

    /**
     * @brief Calculate the symmetric absolute noise immunity external criterion for the given data
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param bufferValues Temporary storage for calculated coefficients and target values 
     * @return The value of the symmetric absolute noise immunity external criterion and calculated model coefficients 
     */
    PairDVXd symAbsoluteNoiseImmunity(const MatrixXd& xTrain, const MatrixXd& xTest, const VectorXd& yTrain, const VectorXd& yTest,
                                  BufferValues& bufferValues) const;

    /**
     * @brief Get k models from the given ones with the best values of the external criterion
     * 
     * @param combinations Vector of the trained models
     * @param data Object containing parts of a split dataset used in model training. Parameter is used in sequential criterion
     * @param func Function returning the new X train and X test data constructed from the original data using given combination of input variables column indexes. Parameter is used in sequential criterion
     * @param k Number of best models
     * @return Vector containing k best models
     */
    virtual VectorC getBestCombinations(VectorC& combinations, const SplittedData& data, const std::function<MatrixXd(const MatrixXd&, const VectorU16&)> func, int k) const;

    /**
     * @brief Calculate the value of the selected external criterion for the given data.
     * 
     * For the individual criterion this method only calls the getResult() method
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @return The value of the external criterion and calculated model coefficients 
     */
    virtual PairDVXd calculate(const MatrixXd& xTrain, const MatrixXd& xTest, 
                               const VectorXd& yTrain, const VectorXd& yTest) const;

public:
    /// @brief Construct a new Criterion object
    Criterion() {};

    /**
     * @brief Construct a new Criterion object
     * 
     * @param _criterionType Selected external criterion type
     * @param _solver Selected method for linear equations solving
     */
    Criterion(CriterionType _criterionType, Solver _solver = Solver::balanced);

    friend class GmdhModel;
};

/// @brief Class that implements calculations of parallel external criterions
class GMDH_API ParallelCriterion : public Criterion {
    CriterionType secondCriterionType; //!< Selected second CriterionType object
    double alpha; //!< coefficient in [0,1] range determining the contribution of the first external criterion to the final result

    /**
     * @brief Calculate the value of the selected parallel external criterion for the given data.
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @return The value of the parallel external criterion and calculated model coefficients  
     */
    PairDVXd calculate(const MatrixXd& xTrain, const MatrixXd& xTest, 
                       const VectorXd& yTrain, const VectorXd& yTest) const override;
public:
    /**
     * @brief Construct a new ParallelCriterion object
     * 
     * @param _firstCriterionType Selected first individual external criterion type
     * @param _secondCriterionType Selected second individual external criterion type
     * @param _alpha Coefficient in [0,1] range determining the contribution of the first external criterion to the final result
     * @param _solver Selected method for linear equations solving
     * @throw std::invalid_argument
     */
    ParallelCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, 
                        double _alpha = 0.5, Solver _solver = Solver::balanced);
};

/// @brief Class that implements calculations of sequential external criterions
class GMDH_API SequentialCriterion : public Criterion {
    CriterionType secondCriterionType; //!< Selected second CriterionType object
    int top; //!< Number of models that should remain after applying the first individual external criterion

    /**
     * @brief Calculate the second external criterion for the remaining models
     * 
     * @param xTrain Input variables matrix of the training data
     * @param xTest Input variables matrix of the testing data
     * @param yTrain Target values vector of the training data
     * @param yTest Target values vector of the testing data
     * @param _coeffsTrain Coefficients vector calculated using training data obtained during the calculation of the first criterion
     * @return The final value of the sequential external criterion and calculated model coefficients
     */
    PairDVXd recalculate(const MatrixXd& xTrain, const MatrixXd& xTest,
        const VectorXd& yTrain, const VectorXd& yTest, const VectorXd& _coeffsTrain) const;
    
    /**
     * @brief Get k models from the given ones with the best values of the sequential external criterion
     * 
     * @param combinations Vector of the trained models
     * @param data Object containing parts of a split dataset used in model training
     * @param func Function returning the new X train and X test data constructed from the original data using given combination of input variables column indexes
     * @param k Number of best models
     * @return VectorC Vector containing k best models
     * 
     */
    VectorC getBestCombinations(VectorC& combinations, const SplittedData& data, const std::function<MatrixXd(const MatrixXd&, const VectorU16&)> func, int k) const override;
public:
    /**
     * @brief Construct a new SequentialCriterion object
     * 
     * @param _firstCriterionType Selected first individual external criterion type
     * @param _secondCriterionType Selected second individual external criterion type
     * @param _top Number of models that should remain after applying the first individual external criterion
     * @param _solver Selected method for linear equations solving
     * @throw std::invalid_argument
     */
    SequentialCriterion(CriterionType _firstCriterionType, CriterionType _secondCriterionType, 
                        int _top=0, Solver _solver = Solver::balanced);
};
}