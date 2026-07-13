@AbapCatalog.sqlViewName: 'ZV_DUP_INV'

@AccessControl.authorizationCheck: #NOT_REQUIRED

@EndUserText.label: 'Duplicate Invoice Detection'

@UI.headerInfo: { typeName: 'Duplicate Invoice',
typeNamePlural: 'Duplicate Invoices',
title: { type: #STANDARD, value: 'ExternalInvoiceNumber' },
description: { type: #STANDARD, value: 'VendorName' } }

@VDM.viewType: #CONSUMPTION

// OData V2 auto-publish
// Generates service: ZI_DUPLICATE_INVOICE_DETECT_CDS
// Register in SAP GUI: /IWFND/MAINT_SERVICE → Add Service → Alias: Local

define view ZAI_DUPL_INV1
with parameters
// budat is a named SAP data element of type DATS — OData compatible
//@Consumption.defaultValue: '00000000'
@EndUserText.label: 'Posting Date From'
p_budat_from : budat,

//@Consumption.defaultValue: '99991231'
@EndUserText.label: 'Posting Date To'
p_budat_to : budat

as select from bseg as Item

// BKPF — Accounting Document Header
inner join bkpf as Header
on Item.mandt = Header.mandt
and Item.bukrs = Header.bukrs
and Item.belnr = Header.belnr
and Item.gjahr = Header.gjahr

// BSEG self-join — find other vendor lines with same vendor + account type
inner join bseg as DupItem
on DupItem.mandt = Item.mandt
and DupItem.bukrs = Item.bukrs
and DupItem.lifnr = Item.lifnr -- lifnr from bseg
and DupItem.koart = Item.koart

// BKPF self-join — match on ext invoice no + currency + doc type
inner join bkpf as DupHeader
on DupHeader.mandt = DupItem.mandt
and DupHeader.bukrs = DupItem.bukrs
and DupHeader.belnr = DupItem.belnr
and DupHeader.gjahr = DupItem.gjahr
and DupHeader.xblnr = Header.xblnr -- same external invoice number
and DupHeader.waers = Header.waers -- same currency
and DupHeader.blart = Header.blart -- same document type
and DupHeader.stblg = '' -- exclude reversals on dup side

// LFA1 — Vendor Master for name
left outer join lfa1 as Vendor
on Vendor.mandt = Item.mandt
and Vendor.lifnr = Item.lifnr

// T001 — Company Code description
left outer join t001 as CompCode
on CompCode.mandt = Header.mandt
and CompCode.bukrs = Header.bukrs

{
// -- Keys ---------------------------------------------------
@ObjectModel.text.element: [ 'CompanyCodeName' ]
@UI.lineItem: [ { position: 10, label: 'Company Code' } ]
@UI.selectionField: [ { position: 10 } ]
key Header.bukrs as CompanyCode,

@UI.lineItem: [ { position: 20, label: 'Document Number' } ]
key Header.belnr as InvoiceDocNumber,

@UI.lineItem: [ { position: 30, label: 'Fiscal Year' } ]
@UI.selectionField: [ { position: 20 } ]
key Header.gjahr as FiscalYear,

key Item.buzei as LineItem,

// -- Header Fields ------------------------------------------
@Search.defaultSearchElement: true
@UI.lineItem: [ { position: 40, label: 'Vendor Invoice No.' } ]
@UI.selectionField: [ { position: 30 } ]
Header.xblnr as ExternalInvoiceNumber,

@ObjectModel.text.element: [ 'VendorName' ]
@UI.lineItem: [ { position: 50, label: 'Vendor' } ]
@UI.selectionField: [ { position: 40 } ]
Item.lifnr as Vendor,

@UI.hidden: true
Vendor.name1 as VendorName,

@UI.hidden: true
CompCode.butxt as CompanyCodeName,

@UI.lineItem: [ { position: 60, label: 'Document Type' } ]
Header.blart as DocumentType,

@UI.lineItem: [ { position: 70, label: 'Posting Date' } ]
@UI.selectionField: [ { position: 50 } ]
Header.budat as PostingDate,

@UI.lineItem: [ { position: 80, label: 'Document Date' } ]
Header.bldat as DocumentDate,

@UI.lineItem: [ { position: 90, label: 'Currency' } ]
Header.waers as Currency,

@Semantics.amount.currencyCode: 'Currency'
@UI.lineItem: [ { position: 100, label: 'Invoice Amount' } ]
Item.wrbtr as InvoiceAmount,

Header.bktxt as HeaderText,
Header.usnam as CreatedBy,
Header.cpudt as InvoiceCreationDate,

// -- Duplicate Count ----------------------------------------
// COUNT(*) is valid in SELECT list
// Aggregate functions NOT allowed inside CASE WHEN in CDS
@UI.lineItem: [ { position: 110, label: 'Duplicate Count', criticality: 'DuplicateCriticality' } ]
count(*) as DuplicateCount,

// -- Criticality --------------------------------------------
// cast literal used — CASE WHEN count(*) not allowed in CDS
// For full traffic-light logic wrap this in a consumption view:
// CASE WHEN DuplicateCount = 2 THEN 2
// WHEN DuplicateCount > 2 THEN 3
// ELSE 1 END
@UI.hidden: true
cast(2 as abap.int1) as DuplicateCriticality,

// -- Line Item Fields ---------------------------------------
Item.hkont as GLAccount,
Item.kostl as CostCenter,
Item.aufnr as OrderNumber,
Item.zuonr as AssignmentNumber,
Item.sgtxt as ItemText
}

where
// Parameter-driven filters
Header.budat >= $parameters.p_budat_from -- Posting Date From
and Header.budat <= $parameters.p_budat_to -- Posting Date To
// Static filters — always applied
and Header.blart = 'RE' -- Vendor Invoice document type only
and Header.xblnr <> '' -- External invoice number must exist
and Item.lifnr <> '' -- Vendor must exist (from bseg)
and Header.stblg = '' -- Exclude reversed documents
and Item.koart = 'K' -- Vendor (Kreditor) line items only

group by Header.bukrs,
Header.belnr,
Header.gjahr,
Item.buzei,
Header.xblnr,
Item.lifnr,
Vendor.name1,
CompCode.butxt,
Header.blart,
Header.budat,
Header.bldat,
Header.waers,
Item.wrbtr,
Header.bktxt,
Header.usnam,
Header.cpudt,
Item.hkont,
Item.kostl,
Item.aufnr,
Item.zuonr,
Item.sgtxt

having count(*) > 1 -- Only return invoices with duplicates
