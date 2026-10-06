"""Render the verified ledger without adding unverified references."""
from utils import *
def main():
    refs=read(ROOT/'sources/citation_verification.json')['references']
    lines=['# 第二階段引用核對','',
      '本次逐項核對一手來源的題名、作者或發布組織、年份及其依據、URL／DOI，以及本文保留主張的支持範圍。論文限已閱讀摘要與出版 metadata；模型限 pinned model card、官方程式與發布身分。未把廠商 benchmark 數字當成本研究結果。','',
      '|ID|Title|Authors / organization|Year / basis|Primary URL / DOI|支持範圍|','|---|---|---|---|---|---|']
    for r in refs:
        authors=', '.join(r.get('authors',[])) or r.get('organization','');year=str(r['year'])+'; '+str(r.get('year_basis','original publication metadata'))
        url='[一手來源]('+r['url']+')'+('; DOI '+r['doi'] if r.get('doi') else '')
        claim=r['supported_claim'].replace('sigma.4?.1','sigma0.4 to0.1')
        lines.append('|'+ '|'.join([r['id'],r['title'],authors,year,url,claim])+'|')
    lines += ['', 'ARC 與 Ovadia 的本機 urllib 下載曾因 TLS 憑證鏈失敗；錯誤原樣保存，未關閉 TLS 檢查。同輪 web 工具已實際開啟一手摘要與 metadata，因此僅在該已閱讀範圍內保留引用。', '',
      'ARC、CommonsenseQA、OpenBookQA 的資料卡已各自核對 schema／split 與授權欄，不再把 OpenBookQA 的授權說明複製到其他資料集。模型年份依 repository creation metadata；不是推定論文發表年。', '',
      '這是 AI 執行的來源檢查，研究者尚未親自覆核。完整檢查狀態、已讀範圍、取得時間、錯誤與本機證據在 `sources/citation_verification.json`；最終報告不引用未驗證的 benchmark、DOI 或外部錯題事實。']
    write(ROOT/'reports/citation_verification.md','\n'.join(lines)+'\n');write(ROOT/'reports/citation_check_v2.md','\n'.join(lines)+'\n')
if __name__=='__main__':main()
