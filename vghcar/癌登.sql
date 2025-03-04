SELECT
    CONCAT(HEX(RAND()), HEX(RAND())) AS LV_UUID,
    PBAB.PDESC     AS PBAB_PDESC    ,
    PBAS.PLSTVDT   AS PBAS_PLSTVDT  ,
    PBAS.PMEDSTAT  AS PBAS_PMEDSTAT ,
    PBC.PGROUP1    AS PBC_PGROUP1   ,
    PBC.TFLAG      AS PBC_TFLAG     ,
    PBC.TDATE      AS PBC_TDATE     ,
    PBC.NFLAG      AS PBC_NFLAG     ,
    PBC.NDATE      AS PBC_NDATE     ,
    PBC.MFLAG      AS PBC_MFLAG     ,
    PBC.MDATE      AS PBC_MDATE     ,
    PBC.DFSDATE    AS PBC_DFSDATE   ,
    PBC.PB3MDATE   AS PBC_PB3MDATE  , -- 最初病理診斷日期 (1st)
    PBC.PB3TDATE   AS PBC_PB3TDATE  , -- 最初臨床診斷日期 (2nd)
    TCDB.DOFC      AS TCDB_DOFC    ,
    TCDB.CISTCSER  AS TCDB_CISTCSER ,
    TCDB.DATATYPE  AS TCDB_DATATYPE ,
    TCDB.PHISTNUM  AS TCDB_PHISTNUM ,
    TCDB.SEX       AS TCDB_SEX      ,
    TCDB.DISAGE    AS TCDB_DISAGE   ,
    TCDB.PRIST     AS TCDB_PRIST    ,
    TCDB.HISTGY    AS TCDB_HISTGY   ,
    TCDB.PSG       AS TCDB_PSG      ,
    TCDB.CLG       AS TCDB_CLG      ,
    TCDB.PSD       AS TCDB_PSD      ,
    TCDB.CLT       AS TCDB_CLT      ,
    TCDB.CLN       AS TCDB_CLN      ,
    TCDB.CLM       AS TCDB_CLM      ,
    TCDB.VOTHSTG   AS TCDB_VOTHSTG  ,
    TCDB.COTHSTG   AS TCDB_COTHSTG  ,
    TCDB.DOID      AS TCDB_DOID     , -- 最初診斷日期     (3rd)
    TCDB.DLCOD     AS TCDB_DLCOD    ,
    TCDB.COAOD     AS TCDB_COAOD    ,
    TCDB.VITSS     AS TCDB_VITSS
    FROM        VGHTPEVG.PBASINFO AS PBAS, VGHTPEVG.CISTCRM AS TCDB
    LEFT JOIN   VGHTPEVG.PBCANCER AS PBC
         ON      PBC.PHISTNUM = TCDB.PHISTNUM
         AND     PBC.PBCSEQNO = TCDB.PBCSEQNO
         AND     PBC.PBCFLAG = 1 -- 有效紀錄 Y1N0 -9DEFAULT 1
         AND     (PBC.PTRTST01 in ('1', '2') OR TCDB.CLASS in ('1', '2')) -- 本院治療 -- 個案分類代碼
         AND     NOT ((PBC.PBCRSTUS = '2' OR PBC.PBCRSTUS = '1') AND PBC.PBCFLAG = 0) AND PBC.PBCFLAG!=2
         AND     (SUBSTR(PBC.PBICD9,1,3) != '225' AND SUBSTR(PBC.PBICD9,1,3) != '227') -- 良性腫瘤
         AND     (PBC.PBCSEQPN != 0)
         AND     SUBSTR(PBICDO3M,7,1) != '0'
    LEFT JOIN (
        (
            SELECT PHISTNUM, PDESC FROM VGHTPEVG.PBABSTRC 
            WHERE PCATG = 'GG' 
            AND PEDOCID LIKE 'CTC%'
        ) -- 國健署回報檔
            UNION
        (
            SELECT PHISTNUM, PDESC FROM VGHTPEVG.PBABSTRC 
            WHERE PCATG = 'GG' 
            AND PEDOCID LIKE 'DOC%'
            AND PHISTNUM NOT IN (SELECT PHISTNUM FROM VGHTPEVG.PBABSTRC WHERE PCATG = 'GG' AND PEDOCID LIKE 'CTC%')
        ) -- 醫師回報資料，去掉國健署的回報檔
    ) PBAB
        ON       PBAB.PHISTNUM = TCDB.PHISTNUM
    WHERE
        TCDB.DATATYPE in ('1', '4') -- 資料型態為原始資料
        AND TCDB.DELF = 'N'
        AND TCDB.PHISTNUM = PBAS.PHISTNUM
        AND TCDB.PHISTNUM != '113'
        AND TCDB.CLASS in ('1', '2') -- 本院治療 -- 個案分類代碼
        AND TCDB.SEQNO is not NULL -- 有效紀錄
    WITH UR;
