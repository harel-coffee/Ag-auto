
generate_map_1 <- function(input_f, CodMothParams){
	sub1 = input_f
	rm (input_f)

	sub2 = sub1[, .(RelPctDiap = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF, RelLarvaPop))*100, RelPctNonDiap = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100, AbsPctDiap = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100, AbsPctNonDiap = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")]
	
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(RelPctDiapGen1 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(RelPctDiapGen2 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(RelPctDiapGen3 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(RelPctDiapGen4 = (auc(CumulativeDDF,RelDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(RelPctNonDiapGen1 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(RelPctNonDiapGen2 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(RelPctNonDiapGen3 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(RelPctNonDiapGen4 = (auc(CumulativeDDF,RelNonDiap)/auc(CumulativeDDF,RelLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	#
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(AbsPctDiapGen1 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(AbsPctDiapGen2 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(AbsPctDiapGen3 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(AbsPctDiapGen4 = (auc(CumulativeDDF,AbsDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	#
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[5,5] & CumulativeDDF < CodMothParams[5,6], .(AbsPctNonDiapGen1 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[6,5] & CumulativeDDF < CodMothParams[6,6], .(AbsPctNonDiapGen2 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[7,5] & CumulativeDDF < CodMothParams[7,6], .(AbsPctNonDiapGen3 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	sub2 = merge(sub2, sub1[CumulativeDDF >= CodMothParams[8,5] & CumulativeDDF < CodMothParams[8,6], .(AbsPctNonDiapGen4 = (auc(CumulativeDDF,AbsNonDiap)/auc(CumulativeDDF,AbsLarvaPop))*100), by = c("ClimateGroup", "CountyGroup", "latitude", "longitude")], by = c("ClimateGroup", "CountyGroup", "latitude", "longitude"), all.x = TRUE)
	

    return (sub2)
}
