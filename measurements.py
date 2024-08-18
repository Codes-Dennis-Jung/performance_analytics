import pandas as pd				
import numpy as np				
import scipy				
from scipy.stats import norm,t, jarque_bera
from scipy import stats				
import statsmodels.api as sm				
import warnings
warnings.filterwarnings('ignore')

def __check_clean_data(data):				
    """ Check data """				
    if isinstance(data,pd.Series):				
        data=data.to_frame()				
    elif isinstance(data,pd.DataFrame):				
        data				
    else:				
        raise "Input data are not a pandas DataFrame or Series!"				
    return data				
				
def __scaling(scale):				
    """ Scaling parameter setting """				
    try:				
        if scale=="daily":				
            scale=252				
        elif scale=="weekly":				
            scale=52				
        elif scale=="monthly":				
            scale=12				
        elif scale=="quarterly": 				
            scale=4 				
        elif scale=="yearly": 				
            scale=1				
    except KeyError:				
        raise ValueError("Please insert correct scaling!")				
    return scale				
				
def __percent(percent,series):				
    if percent:				
        series=series*100								
    return series				
				
def return_calculate(data,discret=True,
                    percent=True):				
    """ Compute discrete or continous returns """				
    df_=__check_clean_data(data)
    if discret: 				
        calculate_return=(df_/df_.shift(1))-1	
        calculate_return.iloc[0,:]=0				
    else:				
        calculate_return=np.log(df_).diff()				
        calculate_return.iloc[0,:]=0				
    return __percent(percent,calculate_return)				

def return_annualized(data,scale="monthly",
                    geometric=True,percent=True):				
    """ Compute annualized returns """				
    df_=__check_clean_data(data)				
    sc_=__scaling(scale)				
    n=len(df_)				
    if geometric:				
        ann_ret=(df_+1).prod(axis=0,skipna=True)**(sc_/n)-1				
    else:				
        ann_ret=df_.mean(axis=0,skipna=True)*sc_				
    an_ret=pd.DataFrame(__percent(percent,ann_ret),
                        index=df_.columns,
                        columns=['Annualized Return'])				
    return an_ret				
				
def return_annualized_excess(data,bmk,
                            scale="monthly",
                            geometric=True,percent=True):					
    """ Compute annualized excess returns """				
    df_=__check_clean_data(data)				
    bm_=__check_clean_data(bmk)				
    n=len(df_.index)				
    m=len(bm_.index)				
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
    
    df_ret=return_annualized(df_,scale,
                            geometric,percent=False)				
    bm_ret=return_annualized(bm_,scale,
                            geometric,percent=False)				
    if geometric:				
        annualized_excess_return=pd.DataFrame((1+ df_ret.values) /\
            (1+bm_ret.values)-1,index=df_.columns)				
    else:				
        annualized_excess_return=pd.DataFrame(df_ret.values\
           -bm_ret.values,index=df_.columns)				
    an_ret=pd.DataFrame((__percent(percent,annualized_excess_return)).values,
                        index=df_.columns,columns=['Annualized Excess Return'])				
    return an_ret				
				
def return_cumulative(data,percent=True):				
    """ Compute cumulative returns """				
    df_=__check_clean_data(data)
    df_.iloc[0,:]=0
    cumulative_ret=(1+df_).cumprod(0)-1					
    cum_ret=pd.DataFrame((__percent(percent,cumulative_ret)).values,
                        index=cumulative_ret.index,
                        columns=df_.columns) 	
    if percent:
        cum_ret=cum_ret+100
    else:
        cum_ret=cum_ret+1	
    return cum_ret

def return_cumulative_zero(data,percent=True):				
    """ Compute cumulative returns """				
    df_=__check_clean_data(data)
    df_.iloc[0,:]=0
    cumulative_ret=(1+df_).cumprod(0)-1					
    cum_ret=pd.DataFrame((__percent(percent,cumulative_ret)).values,
                        index=cumulative_ret.index,
                        columns=df_.columns) 					
    return cum_ret				
				
def return_excess(data,bmk,percent=True):				
    """ Compute excess return time series """				
    df_=__check_clean_data(data)				
    bm_=__check_clean_data(bmk)				
    n=len(df_.index)				
    m=len(bm_.index)				
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
    excess_return=pd.DataFrame(df_.values-bm_.values,
                                index=df_.index,
                                columns=df_.columns)				
    excess_return.iloc[0,:]=0				
    excess_return=pd.DataFrame((__percent(percent,excess_return)).values,
                                index=excess_return.index,
                                columns=excess_return.columns) 				
    return excess_return				
				
def significance_test(data,bmk):				
    """ Compute significance test """				
    df_=__check_clean_data(data)				
    bm_=__check_clean_data(bmk)				
    n=len(df_.index)				
    m=len(bm_.index)				
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
	
    _Signif=[]				
    for k in range(0,len(list(df_))):				
        f=df_.iloc[:,[k]].values-bm_.values				
        x=np.ones(np.shape(f)) 				
        model=sm.OLS(f,x,missing='drop') 				
        results=model.fit(cov_type='HAC',
                            cov_kwds={'maxlags': 6}) 				
        r=np.zeros_like(results.params)				
        r[:]=[1]				
        res=results.t_test(r).pvalue                				
        _Signif.append(pd.DataFrame(np.array([res]),
                                columns=df_.iloc[:,[k]].columns,
                                index=['P-value']))				
    res=pd.concat(_Signif,sort=False,axis=1)				
    return res				
	
def false_discovery_control(pval):				
    """ 				
        Compute false discovery test from Benjamini Yekutieli				
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html				
    """				
    df_=__check_clean_data(pval)				
    res=stats.false_discovery_controls(df_.to_numpy(),
                                        method='by')				
    res=pd.Dataframe(res,index=df_.index,
                    columns=df_.columns)				
    return res				
				
def seriel_correlation(data): 				
    """ Compute seriel correlation """				
    df_=__check_clean_data(data)				
    AutoCorr=[]				
    for i in range(0,len(list(df_))):				
        series=df_.iloc[:,[i]].squeeze()				
        ser1=series.autocorr(lag=1)				
        ser2=series.autocorr(lag=2)				
        ser3=series.autocorr(lag=3)				
        ser4=series.autocorr(lag=4)				
        ser5=series.autocorr(lag=5)				
        SerielCorrel=pd.DataFrame([ser1,ser2,ser3,ser4,ser5],				
                                    index=['AR(1)',
                                            'AR(2)',
                                            'AR(3)',
                                            'AR(4)',
                                            'AR(5)'],				
                                    columns=[series.name])				
        AutoCorr.append(SerielCorrel)				
    serial_correlation=pd.concat(AutoCorr,axis=1)    				
    return serial_correlation				
				
def signal_to_noise_ratio(data,signal):	
    """ Compute signal to noise ratio """				
    df_=__check_clean_data(data)				
    sig_=__check_clean_data(signal)				
    n=len(df_.index)				
    m=len(sig_.index)				
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
    lst=[] 				
    for i in range(0,len(list(df_))):				
        nom=df_.iloc[:,[i]].corrwith(method='pearson',
                                    other=sig_.iloc[:,[i]],
                                    axis=0)				
        denom=np.sqrt(1-nom.values**2)				
        SNR=nom.values/denom				
        lst.append(SNR)				
    res=pd.concat(lst,axis=1)				
    res.columns=df_.columns				
    res.index=['Signal to Noise']				
    return SNR				
				
def dm_test(actual_df,pred1_df,pred2_df,h,crit,power=None):	
    """ Compute DM test """			
    actual_lst=__check_clean_data(actual_df).iloc[:,[0]].\
        squeeze('columns').tolist()				
    pred1_lst=__check_clean_data(pred1_df).iloc[:,[0]].\
        squeeze('columns').tolist()				
    pred2_lst=__check_clean_data(pred2_df).iloc[:,[0]].\
        squeeze('columns').tolist()				
    e1_lst=[]				
    e2_lst=[]				
    d_lst=[]				
    T=float(len(actual_lst))				
				
    if crit=="MSE":				
        for actual,p1,p2 in zip(actual_lst,
                                pred1_lst,pred2_lst):				
            e1_lst.append((actual-p1)**2)				
            e2_lst.append((actual-p2)**2)				
        for e1,e2 in zip(e1_lst,e2_lst):				
            d_lst.append(e1-e2)				
    elif crit=="MAD":				
        for actual,p1,p2 in zip(actual_lst,
                                pred1_lst,pred2_lst):				
            e1_lst.append(abs(actual-p1))				
            e2_lst.append(abs(actual-p2))				
        for e1,e2 in zip(e1_lst,e2_lst):				
            d_lst.append(e1-e2)				
    elif crit=="MAPE":				
        for actual,p1,p2 in zip(actual_lst,
                                pred1_lst,pred2_lst):				
            e1_lst.append(abs((actual-p1)/actual))				
            e2_lst.append(abs((actual-p2)/actual))				
        for e1,e2 in zip(e1_lst,e2_lst):				
            d_lst.append(e1-e2)				
    elif crit=="poly":				
        for actual,p1,p2 in zip(actual_lst,
                                pred1_lst,pred2_lst):				
            e1_lst.append(((actual-p1))**(power))				
            e2_lst.append(((actual-p2))**(power))				
        for e1,e2 in zip(e1_lst,e2_lst):				
            d_lst.append(e1-e2)				
				
    mean_d=pd.Series(d_lst).mean()				
				
    def autocovariance(Xi,N,k,Xs):				
        autoCov=0				
        T=float(N)				
        for i in np.arange(0,N-k):				
            autoCov+=((Xi[i+k])-Xs)*(Xi[i]-Xs)				
        return (1/(T))*autoCov				
				
    gamma=[]				
    for lag in range(0,h):				
        gamma.append(autocovariance(d_lst,
                                    len(d_lst),
                                    lag,mean_d))				
    V_d=(gamma[0]+2*sum(gamma[1:]))/T				
    DM_stat=V_d**(-0.5)*mean_d				
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)				
    DM_stat=harvey_adj*DM_stat				
    p_value=2*t.cdf(-abs(DM_stat),df=T-1)				
    return p_value				

def skewness_annualized(data,scale ='monthly'):				
    """ Compute annualized skewness """				
    df_=__check_clean_data(data)
    sc_=__scaling(scale)		
    skew_an=df_.skew(skipna=True)/np.sqrt([sc_])
    skew_an=skew_an.to_frame()
    skew_an.columns=['Skewness p.a.']
    return skew_an			
				
def kurtosis_annualized(data,scale='monthly'):				
    """ Compute annualized kurtosis """				
    if scale!='monthly':				
        print('Scaling needs to be adjusted!')				
    df_=__check_clean_data(data)				
    kurt_an=df_.kurt(skipna=True)/12+11/4
    kurt_an=kurt_an.to_frame()
    kurt_an.columns=['Kurtosis p.a.']			
    return kurt_an				

def drawdowns(data,geometric=True,percent=True):				
    """ Compute drawdonw time series """				
    df_=__check_clean_data(data)				
    if geometric:				
        cumulative_ret=(1+data.fillna(0)).cumprod(skipna=True)				
        cumulative_ret_max=cumulative_ret.cummax(skipna=True)			
        drawdowns=cumulative_ret.div(cumulative_ret_max,axis=0) -1				
        drawdowns.iloc[0,:]=0				
    else:				
        cumulative_ret=data.fillna(0).cumsum()+1				
        cumulative_ret_max=cumulative_ret.cummax(skipna=True)		
        drawdowns=cumulative_ret.div(cumulative_ret_max,axis=0) -1				
        drawdowns.iloc[0,:]=0				
    drawdn=pd.DataFrame((__percent(percent,drawdowns)).values,
                        index=df_.index,columns=df_.columns) 
    return drawdn	

def beta_coeff(y,x): 
    """ Compute beta coefficient """								
    y=__check_clean_data(y)
    x=__check_clean_data(x)
    if len(y) < 2 or len(x) < 2:				
        msg="Beta is not well-defined with less than two samples."				
        warnings.warn(msg)				
        return float('nan')
    else:				
        lst=[]    				
        for k in range(0,len(list(x))):				
            ind=x.iloc[:,[k]]				
            exog=sm.add_constant(ind)				
            model=sm.OLS(y,exog,missing='drop',
                        hasconst=True) 				
            results=model.fit(cov_type='HAC',
                            cov_kwds={'maxlags': 6}) 				
            beta_coeff=pd.DataFrame(results.params[1],				
                                index=ind.columns,				
                                columns=['Beta coefficient']).round(3)				
            pval=pd.DataFrame(results.pvalues[1],				
                                index=ind.columns,				
                                columns=['P-value']).round(3)				
            lst.append(pd.concat([beta_coeff,pval],axis=1))				
    res= pd.concat(lst,axis=0)
    beta=res.iloc[:,[0]]
    sign=res.iloc[:,[1]]				
    sign=sign.where(cond=sign > 0.01,other=77)				
    sign=sign.where(cond=sign > 0.05,other=88)				
    sign=sign.where(cond=sign > 0.1,other=99)				
    sign=sign.where(cond=sign >= 77,other=1)				
    sign=sign.replace(99,'*').replace(88,'**').replace(77,'***').replace(1,'')				
    sig_beta=pd.concat([beta.round(2),sign],axis=1)
    sig_beta.index = [y.columns]		
    return sig_beta			
				
def max_drawdowns(data,percent=True,geometric=True):
    """ Compute drawdown time series """				
    df_=__check_clean_data(data)				
    dd_max=drawdowns(df_,geometric,percent).min(axis=0,skipna=True)				
    drawdown_max=pd.DataFrame(dd_max,index=dd_max.index,
                                columns=['Maximum Drawdown'])	
    return drawdown_max				
				
def volatility_annualized(data,scale="monthly",percent=True):				
    """ Compute annualized volatility """				
    df_=__check_clean_data(data)				
    sc_=__scaling(scale)				
    annualized_volatility=np.sqrt(sc_)*df_.std(ddof=0,axis=0,
                                               skipna=True,
                                               numeric_only=True)				
    annualized_volatility=pd.DataFrame((__percent(percent,
                                                    annualized_volatility)).values,	
                                        index=annualized_volatility.index,
                                        columns=['Annualized Volatility']) 
    return annualized_volatility				
				
def semideviation(data,scale='monthly',percent=True):
    """ Compute annualized semideviation """				
    df_=__check_clean_data(data)				
    sc_=__scaling(scale)	
    is_negetive= df_<=0
    semi_dev=df_[is_negetive].std(ddof=0,axis=0,
                                  skipna=True,
                                  numeric_only=True)*np.sqrt(sc_)
    semi_dev=semi_dev.to_frame()
    semi_dev.columns=['Annualized Semideviation']
    return __percent(percent,semi_dev)	

def coskewness(data_bmk,bias_=True):
    """ Compute coskewness """
    df=__check_clean_data(data_bmk)				
    v=df.values
    s1=sigma=v.std(0,keepdims=True)
    means=v.mean(0,keepdims=True)
    v1=v-means
    s2=sigma**2
    v2=v1**2
    m=v.shape[0]
    skew=pd.DataFrame(v2.T.dot(v1)/s2.T.dot(s1)/m,df.columns,df.columns)
    if bias_:
        skew *= ((m-1)*m)**.5/(m-2)
    skew=skew.iloc[[0],[-1]]
    skew.columns=['Coskewness']
    return skew

def cokurtosis(data_bmk,bias_=True,fisher=True,variant='middle'):
    """ Compute cokurtosis """
    df=__check_clean_data(data_bmk)
    v=df.values
    s1=sigma=v.std(0,keepdims=True)
    means=v.mean(0,keepdims=True)
    v1=v-means
    s2=sigma**2
    s3=sigma**3
    v2=v1**2
    v3=v1**3
    m=v.shape[0]
    if variant in ['left','right']:
        kurt=pd.DataFrame(v3.T.dot(v1)/s3.T.dot(s1)/m,df.columns,df.columns)
        if variant=='right':
            kurt=kurt.T
    elif variant=='middle':
        kurt=pd.DataFrame(v2.T.dot(v2)/s2.T.dot(s2)/m,df.columns,df.columns)
    if bias_:
        kurt=kurt*(m**2-1)/(m-2)/(m-3)-3*(m-1)**2/(m-2)/(m-3)
    if not fisher:
        kurt+=3
    kurt=kurt.iloc[[0],[-1]]
    kurt.columns=['Cokurtosis']
    return kurt

def tracking_error(data,bmk,scale="monthly",percent=True):				
    """ Compute tracking error """				
    df_=__check_clean_data(data)				
    bm_=__check_clean_data(bmk)	
    sc_=__scaling(scale)			
    n=len(df_.index)
    m=len(bm_.index)			
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
    excess_return=pd.DataFrame(df_.values-bm_.values,
                                index=df_.index)				
    excess_return.iloc[0,:]=0				
    error_tracking=np.sqrt(sc_)*excess_return.std(ddof=0,axis=0,
                                               skipna=True,
                                               numeric_only=True)				
    error_tracking=pd.DataFrame((__percent(percent,
                                            error_tracking)).values,
                                index=df_.columns,
                                columns=['Tracking Error']) 				
    return error_tracking				

def information_ratio(data,bmk,scale="monthly",geometric=True,percent=True):				
    """ Compute information ratio """				
    if geometric:				
        ret_exc=return_annualized_excess(data,bmk,scale,
                                        geometric=True,percent=True)				
        te=tracking_error(data,bmk,scale,percent=True)				
    else:				
        ret_exc=return_annualized_excess(data,bmk,scale,
                                        geometric=False,percent=True)				
        te=tracking_error(data,bmk,scale,percent=True)				
    ratio_information=pd.DataFrame(ret_exc.values/te.values,
                                    index=data.columns,
                                    columns=['Information Ratio'])	
    ir=ratio_information.fillna('Error')			
    return ir				
				
def return_risk_ratio_annualized(data,scale="monthly"):				
    """ Compute sharpe ratio """	
    ret=return_annualized(data,scale,geometric=True,percent=True)				
    vol=volatility_annualized(data,scale,percent=True)				
    rr_ratio=pd.DataFrame(ret.values/vol.values,				
                            index=data.columns,				
                            columns=['Return to Risk Ratio'])
    return rr_ratio				
				
def turnover_avg(weights,percent=True):				
    """ 				
        Compute average turnover 				
        DeMiguel et al. (2009)				
    """				
    df_=__check_clean_data(weights)				
    to=(df_-df_.shift(1))				
    to.iloc[0,:]=0				
    to=to.abs().sum(axis=1).mean(axis=0,skipna=True)*100				
    to=pd.DataFrame(__percent(percent,to),
                    index=['Portfolio'],columns=['Average Turnover']) 				
    return to				
				
def net_return_series(weights,data,TC=0):				
    """ Net returns - Marshall et al. 2012 & Sakkas """				
    w_=__check_clean_data(weights)				
    df_=__check_clean_data(data)				
    n=len(df_.index)				
    m=len(w_.index)				
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
    to=(w_-w_.shift(1))
    to.iloc[0,:]=0				
    rets_2 =TC*(to.abs()).dot(df_)				
    rets_=(df_.dot(w_)).sum()				
    ret=rets_-rets_2
    return ret

def risk_adj_abn_ret(data,bmk,geometric=True,
                    scale="monthly",percent=True):				
    """ Compute risk-adjusted abnormal returns """				
    df_vol=volatility_annualized(data,scale,percent)				
    bmk_vol=volatility_annualized(bmk,scale,percent)				
    exc_ret=return_annualized_excess(data,bmk,
                                    scale,geometric,percent)				
    raar=pd.DataFrame(exc_ret.values*bmk_vol.values/df_vol.values,				
                        index=data.columns,				
                        columns=['Risk-Adjusted Abnormal Return'])				
    return raar	

def sharpe_ratio(data,rfr,geometric=True,
                    scale="monthly",percent=True):				
    """ Compute Sharpe ratio """				
    df_vol=volatility_annualized(data,scale,percent)				
    exc_ret=return_annualized_excess(data,rfr,scale,geometric,percent)
    sr=pd.DataFrame(exc_ret.values/df_vol.values,				
                        index=data.columns,				
                        columns=['Sharpe Ratio'])				
    return sr		

def appraisal_ratio(data,bmk,geometric=True,
                    scale="monthly",percent=True):				
    """ Compute Appraisal ratio """				
    exc_ret=return_annualized_excess(data,bmk,
                                       scale,geometric,percent)	
    te=tracking_error(data,bmk,scale,percent)	
    app_r=pd.DataFrame(exc_ret.values/te.values,				
                        index=data.columns,				
                        columns=['Appraisal Ratio'])				
    return app_r

def appraisal_ratio_regress(data,bmk):				
    """ Compute Treynor ratio """				
    df_=__check_clean_data(data)				
    bm_=__check_clean_data(bmk)				
    n=len(df_.index)				
    m=len(bm_.index)				
    k=len(list(df_))				
    l=len(list(bm_))				
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
    lst=[]		
    for i in range(k):				
        if l > 1:				
            lst.append(sm.OLS(df_.iloc[:,[i]],
                              sm.add_constant(bm_.iloc[:,[i]])).\
                                  fit(cov_type='HAC',cov_kwds={'maxlags': 6}).params[-1])				
        else:				
            res=sm.OLS(df_.iloc[:,[i]],
                              sm.add_constant(bm_)).\
                                  fit(cov_type='HAC',cov_kwds={'maxlags': 6})
            alpha=pd.DataFrame(res.params).T.iloc[:,[0]].values
            se=pd.DataFrame(res.bse).T.iloc[:,[0]].values
            lst.append(pd.DataFrame(alpha/se,columns=[i]))
    res=pd.concat(lst,axis=1)
    res.columns=df_.columns
    res.index=['Appraisal Ratio (alpha/se)']
    return res.T
    
def hit_ratio(data,bmk,percent=True):				
    """ Compute hit ratio """
    df_=__check_clean_data(data)				
    bm_=__check_clean_data(bmk)
    n=len(df_.index)				
    m=len(bm_.index)				
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
    df_dir=pd.DataFrame(np.sign(df_).values).replace(0,1)				
    bmk_dir=pd.DataFrame(np.sign(bm_).values).replace(0,1)				
    hr=pd.DataFrame(np.where((df_dir-bmk_dir)!=0,0,1),
                    index=df_.index,columns=df_.columns).sum()/n				
    hr=pd.DataFrame(__percent(percent,hr),
                    columns=['Hit Ratio']) 				
    return hr
				
def calmar_ratio(data,scale="monthly",				
                geometric=True,percent=True):
    """ Compute calmar ratio """				
    ret=return_annualized(data,scale,geometric,percent)				
    dd=max_drawdowns(data,geometric, percent)				
    cal=pd.DataFrame(ret.values/abs(dd.values),
                    columns=['Calmar Ratio'],				
                    index=data.columns)
    return cal				
				
def value_at_risk(data,cutoff=0.05,percent=True):				
    """ Compute value at risk """				
    df_=__check_clean_data(data)				
    n=len(df_.columns)				
    var=pd.DataFrame([np.percentile(df_.iloc[:,[x]],100*cutoff) for x in range(n)],
                    columns =['VaR {}'.format((1-cutoff)*100)],				
                    index=df_.columns)	
    return __percent(percent,var)				
				
def value_at_risk_cornish_fisher(data,cutoff=0.05,percent=True):				
    """ Compute value at risk with cornish fisher expansion """				
    df_=__check_clean_data(data)				
    k=df_.kurt()				
    s=df_.skew()				
    z=norm.ppf(cutoff)				
    z=(z+(z**2-1)*s/6+(z**3-3*z)*(k-3)/24-(2*z**3-5*z)*(s**2)/36) 						
    var=-(df_.mean()+z*df_.std(ddof=0))
    var=var.to_frame()			
    var.columns=['VaR (Cornish Fisher) {}'.format((1-cutoff)*100)]
    return __percent(percent,-1*var)

def value_at_risk_gaussian(data,cutoff=0.05,percent=True):	
    """  Compute parametric Gaussian VaR """
    df_=__check_clean_data(data)	
    z=norm.ppf(cutoff)
    var=-(df_.mean()+z*df_.std(ddof=0))
    var=var.to_frame()			
    var.columns=['VaR (Gaussian) {}'.format((1-cutoff)*100)]
    return __percent(percent,-1*var)

def is_normal(data,level=0.05):
    """ Applies the Jarque-Bera """
    _,p_value=scipy.stats.jarque_bera(data)
    return p_value > level
				
def conditional_value_at_risk(data,cutoff=0.05,percent=True):				
    """ Compute conditional value at risk """				
    df_=__check_clean_data(data)				
    n=len(df_.columns)				
    lst=[]				
    for i in range(0,n):				
        df_in=df_.iloc[:,i]				
        cutoff_index=int((len(df_in)-1)*cutoff)				
        cvar=np.mean(np.partition(df_in,cutoff_index)[:cutoff_index+1])				
        lst.append(cvar)				
    cvar=pd.DataFrame(lst,columns =['CVaR {}%'.format((1-cutoff)*100)],
                        index=df_.columns)
    return __percent(percent,cvar)				
				
def information_coefficient(data,signal,percent=True):				
    """ Compute IC """				
    df_=__check_clean_data(data)				
    sig_=__check_clean_data(signal)				
    n=len(df_.index)				
    m=len(sig_.index)				
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
    ic=pd.DataFrame([stats.spearmanr(df_.iloc[:,x],sig_.iloc[:,x])[0] \
        for x in range(len(df_.columns))],index=df_.columns,
                    columns=['Information Coefficient'])				
    return __percent(percent,ic)				

def omega_ratio(data,rfr=0.0,required_return=0.0,				
                scale="monthly"):				
    """ Compute Omega ratio """				
    df_=__check_clean_data(data)				
    sc_=__scaling(scale)				
    risk_free=rfr				
    n=len(list(df_))				
    if required_return <= -1:				
        return np.nan				
    elif isinstance(required_return,(pd.DataFrame,pd.Series)):				
        return_threshold=(1+required_return)**(1./sc_)-1				
    else:				
        return_threshold=required_return				
    returns_less_thresh=df_-risk_free-return_threshold				
    lst=[]				
    for i in range(n):				
        numer=returns_less_thresh.iloc[:,[i]][returns_less_thresh.iloc[:,[i]] > 0.0].sum()				
        denom=-1.0*returns_less_thresh.iloc[:,[i]][returns_less_thresh.iloc[:,[i]] < 0.0].sum()				
        lst.append(pd.DataFrame(numer/denom,				
                                columns=["Omega Ratio"]))
    return pd.concat(lst,axis=0)				
				
def sortino_ratio(data,rfr,scale="monthly"):				
    """ Compute Sortino ratio """				
				
    df_=__check_clean_data(data)				
    sc_=__scaling(scale)				
    n=len(list(df_))				
    down_diff=df_-rfr.values.reshape(-1,1)		
    lst=[]				
    for i in range(n):				
        df=down_diff.iloc[:,[i]]				
        down_dev=(df[df < df.mean()].std(axis=0,ddof=0,skipna=True,
                                         numeric_only=True)*np.sqrt(sc_)).values				
        ret=return_annualized(df,scale,geometric=False,				
                                percent=False).values				
        lst.append(pd.DataFrame(ret/down_dev,index=df_.iloc[:,[i]].columns,
                                columns=["Sortino Ratio"]))				
    return pd.concat(lst,axis=0)				
				
def treynor_ratio(data,bmk,geometric=True,percent=True,scale="monthly"):				
    """ Compute Treynor ratio """				
    df_=__check_clean_data(data)				
    bm_=__check_clean_data(bmk)				
    n=len(df_.index)				
    m=len(bm_.index)				
    k=len(list(df_))				
    l=len(list(bm_))				
    if n!=m:				
        raise "Length of DataFrames must be equal!"				
    ret=return_annualized_excess(data,bmk,scale,geometric,percent)				
    lst=[]				
    for i in range(k):				
        if l > 1:				
            lst.append(sm.OLS(df_.iloc[:,[i]],
                            sm.add_constant(bm_.iloc[:,[i]])).\
                                fit(cov_type='HAC',cov_kwds={'maxlags': 6}).params[-1])				
        else:				
            lst.append(sm.OLS(df_.iloc[:,[i]],				
                    sm.add_constant(bm_)).\
                        fit(cov_type='HAC',cov_kwds={'maxlags': 6}).params[-1])				
    trey=pd.DataFrame(ret.values/np.array(lst).reshape(-1,1),index=df_.columns,				
                        columns=["Treynor Ratio"])				
    return trey				
				
def portfolio_vol(weights,covmat):				
    """ Compute portfolio volatility """				
    vol=(weights.T @ covmat @ weights)**0.5
    return vol				

def risk_contribution(weights,data,percent=True):				
    """ Compute risk contribution """				
    df_=__check_clean_data(data)				
    w_=__check_clean_data(weights).T				
    cov_=df_.cov()				
    total_portfolio_var=portfolio_vol(w_,cov_)**2				
    marginal_contrib=cov_ @ w_				
    risk_contrib=np.multiply(marginal_contrib,w_)/total_portfolio_var.values				
    risk_cont=pd.DataFrame((__percent(percent,risk_contrib)).values,
                            index=df_.columns,columns=['Risk contribution'])
    return risk_cont				
				
def calculate_portfolio_var(weights,covar):				
    var=(weights.T @ covar @ weights)				
    return var				
				
def diversification_ratio(weights,data):				
    """ Compute diversification ratio """				
    df_=__check_clean_data(data)				
    w_=__check_clean_data(weights).T				
    cov_=df_.cov()				
    w_vol=np.dot(np.sqrt(np.diag(cov_)),w_)				
    port_vol=calculate_portfolio_var(w_,cov_)				
    diversification_ratio=w_vol/port_vol				
    diver=pd.DataFrame(diversification_ratio.values,
                        index=['Portfolio'],				
                        columns=['Diversification Ratio'])				
    
    return diver				
				
def performance_risk_table(data,bmk,rfr,scale="monthly",
                        geometric=True,percent=True,cutoff=0.05):		
    table=pd.concat([return_annualized(data,scale,geometric,percent),
                    return_annualized_excess(data,bmk,scale,geometric,percent),				
                    max_drawdowns(data,percent,geometric),				
                    volatility_annualized(data,scale,percent),	
                    semideviation(data,scale,percent),	
                    tracking_error(data,bmk,scale,percent),				
                    information_ratio(data,bmk,scale,geometric,percent),
                    sharpe_ratio(data,rfr,geometric,scale,percent),
                    sortino_ratio(data,rfr,scale),
                    calmar_ratio(data,scale,geometric,percent),
                    appraisal_ratio(data,bmk,geometric,scale,percent),
                    appraisal_ratio_regress(data,bmk),
                    return_risk_ratio_annualized(data,scale),	
                    risk_adj_abn_ret(data,bmk,geometric,scale,percent),				
                    value_at_risk(data,cutoff,percent),				
                    conditional_value_at_risk(data,cutoff,percent)],axis= 1)				
    return table

def zscore_scaling(data):
    return pd.DataFrame(stats.zscore(data,axis=1,nan_policy='omit'),
                        columns=data.columns,index=data.index)
    
def scaling(data): 
    lst = []
    for i in range(len(data)):
        series = data.iloc[[i],:].squeeze()
        try:
            temp2 = np.squeeze(series.rank(method='first',
                                         numeric_only=True))
            series.loc[~series.isna()] = (temp2/(1+len(temp2)) - 0.5)
        except ValueError:
            raise ValueError("CHECK DATA")
        lst.append(pd.DataFrame(series).T)
    data = pd.concat(lst)
    data = data.replace([np.inf, -np.inf], np.nan)
    return data
