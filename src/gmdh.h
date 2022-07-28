namespace GMDH {

enum class ExternalCriterion {regularity};  // TODO: define and realise external criterions

class GMDH {
    double externalCriterion() const;
    double calculatePolinomial() const;

public:
    virtual void save() const = 0;
    virtual int load() = 0;
    virtual GMDH& fit() = 0;
    virtual double predict() const = 0;
    
};

class COMBI : public GMDH {

    std::vector<bool> bestPolinom;
    std::vector<double> bestCoeffs;

    std::vector<double> internalCriterion() const;
    

    
        
public:
    

    void save() const override;
    int load() override;
    COMBI& fit() override;
    double predict() const override;
};


}