@AbapCatalog.sqlViewName: 'ZV_PO_CR_APP'
@AbapCatalog.compiler.compareFilter: true
@AccessControl.authorizationCheck: #NOT_REQUIRED
@EndUserText.label: 'PO Creator and Approver Same - SoD'
@VDM.viewType: #CONSUMPTION
@OData.publish: true

define view ZAI_PO_CR_APP
  with parameters
    @EndUserText.label: 'Approval Date From'
    p_budat_from   : budat,
    @EndUserText.label: 'Approval Date To'
    p_budat_to     : budat,
    @EndUserText.label: 'Minimum PO Net Value'
    p_po_value     : netwr_ap

  as select from ekko as POHeader

  // EKPO — PO line items
  inner join ekpo as POItem
    on  POItem.ebeln = POHeader.ebeln

  // CDHDR — Change document header for PO release
  // objectclas = 'EINKBELEG' for purchasing documents
  // tcode ME29N = individual release, ME28 = list release
  // username = the person who performed the release (approver)
  // Narrowed by udate window to reduce CDHDR scan.
  inner join cdhdr as RelHdr
    on  RelHdr.objectid   = POHeader.ebeln
    AND RelHdr.objectclas = 'EINKBELEG'
    AND ( RelHdr.tcode    = 'ME29N'
       OR RelHdr.tcode    = 'ME28' )

  // CDPOS — Change document items
  // tabname = 'EKKO' and fname = 'FRGKE' confirms the
  // release indicator was updated (i.e. PO was released)
  inner join cdpos as RelPos
    on  RelPos.objectclas = RelHdr.objectclas
    AND RelPos.objectid   = RelHdr.objectid
    AND RelPos.changenr   = RelHdr.changenr
    AND RelPos.tabname    = 'EKKO'
    AND RelPos.fname      = 'FRGKE'

  // T024 — Purchasing Group description
  left outer join t024 as PurchGroup
    on  PurchGroup.ekgrp = POHeader.ekgrp

  // ZPO_SOD_WHITELIST — exclude reviewed/whitelisted POs
  // TODO: confirm key field(s) in ZPO_SOD_WHITELIST (assumed: ebeln)
  left outer join zpo_sod_whitelist as Whitelist
    on  Whitelist.ebeln = POHeader.ebeln

{
  key POHeader.ebeln                       as PurchaseOrder,
  key POItem.ebelp                         as POItem,
      // ChangeNumber added to key: a PO item may have multiple release events.
      // Grain = one row per (PO item, release change event).
  key RelHdr.changenr                      as ChangeNumber,

      POHeader.bukrs                       as CompanyCode,
      POHeader.bsart                       as POType,
      POHeader.ernam                       as POCreator,
      POHeader.aedat                       as POCreationDate,
      POHeader.bedat                       as PODocumentDate,
      POHeader.ekgrp                       as PurchasingGroup,
      PurchGroup.eknam                     as PurchasingGroupName,
      POHeader.lifnr                       as Supplier,
      POHeader.waers                       as Currency,
      POHeader.frgke                       as ReleaseIndicator,
      POHeader.frgzu                       as ReleaseStatus,
      POHeader.frggr                       as ReleaseGroup,
      POHeader.frgrl                       as ReleaseStrategy,

      POItem.matnr                         as Material,
      POItem.txz01                         as MaterialDescription,
      POItem.werks                         as Plant,
      POItem.menge                         as POQuantity,
      POItem.meins                         as OrderUnit,
      POItem.netpr                         as NetPrice,
      POItem.netwr                         as NetValue,

      // Change document — approver info
      RelHdr.udate                         as ApprovalDate,
      RelHdr.utime                         as ApprovalTime,
      RelHdr.username                      as Approver,
      RelHdr.tcode                         as ApprovalTCode,
      RelPos.value_new                     as ReleaseIndicatorNew,
      RelPos.value_old                     as ReleaseIndicatorOld,

      // Static criticality literal (3 = high risk SoD violation)
      cast( 3 as abap.int1 )               as RiskCriticality
}

where
      // Time window: filter on APPROVAL date (when SoD event occurred)
      RelHdr.udate          between :p_budat_from and :p_budat_to

      // Only true release events — NEW-VALUE indicates release
      // TODO: confirm release indicator code(s) in customer's release strategy
      //       (commonly 'R' = released / 'F' = final release). For now
      //       exclude empty / reset events.
  AND RelPos.value_new      <> ''

      // Core SoD rule — Creator = Approver
  AND POHeader.ernam        = RelHdr.username

      // Value threshold via parameter (per line)
  AND POItem.netwr          >= :p_po_value

      // Company code scope — EMEA entities
      // TODO: replace with mapping CDS / select-options when available
  AND ( POHeader.bukrs      = '1000'
     OR POHeader.bukrs      = '2000'
     OR POHeader.bukrs      = '3000' )

      // Exclude technical / batch users
  AND POHeader.ernam        <> 'WF-BATCH'
  AND POHeader.ernam        <> 'RFC_USER'

      // Exclude framework orders and stock transport orders
  AND POHeader.bsart        <> 'FO'
  AND POHeader.bsart        <> 'UB'

      // Exclude deletion-flagged items
  AND POItem.loekz          = ''

      // Exclude POs whitelisted / already reviewed
  AND Whitelist.ebeln       IS NULL