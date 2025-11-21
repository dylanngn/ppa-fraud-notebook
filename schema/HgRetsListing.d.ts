/**
 * Schema used internally by Homegate to exchange real estate listing data
 */
export declare class Listing {
    /**
     * All data related to the location of the listing
     */
    address?: Address;
    /**
     * The date at which the listing will be available for purchase
     */
    availableFrom?: string;
    /**
     * Federal statistical office specific information
     */
    bfs?: BfsData;
    /**
     * Building zones intended for residential buildings, those are typically municipal-specific
     * abbreviations.
     */
    buildingZones?: string;
    /**
     * Describes what the purpose of the listing property is. Ordered by least specific to most
     * specific
     */
    categories?: Category[];
    /**
     * Set of features or attributes to better classify what the property can offer to the seeker
     */
    characteristics?: Characteristics;
    /**
     * How the contact inquiry form should be configured on detail page
     */
    contactForm?: ContactForm;
    /**
     * The development state of the property.
     */
    development?: Development;
    externalIds?: ExternalIds;
    /**
     * Unique Homegate-internal identifier of the listing.
     */
    id?: null | string;
    legacy?: Legacy;
    lister?: Lister;
    /**
     * Any fields that are language-specific
     */
    localization?: Localization;
    meta?: Meta;
    /**
     * Data related to New Construction projects
     */
    newConstructionData?: NewConstructionData;
    /**
     * Whether the listing is for sale or for rent
     */
    offerType?: OfferType;
    /**
     * Platforms (web portals) on which the listings should be shown. Formerly known as
     * publishings
     */
    platforms?: string[];
    prices?: Prices;
    /**
     * List of publisher custom fields
     */
    publishers?: Publisher[];
    /**
     * Any value added services that are associated with the listing
     */
    valueAddedServices?: ValueAddedServices;
}
/**
 * All data related to the location of the listing
 *
 * The physical postal address of the lister
 *
 * The physical postal address used to notify lister of any billing or invoice information
 */
export declare class Address {
    /**
     * The two letter country code
     */
    country?: Country;
    /**
     * The geo coordinate representation of the listing address
     */
    geoCoordinates?: GeoCoordinates;
    /**
     * Distances for each geoTag from the edge of the polygon to the listing coordinates.
     */
    geoDistances?: GeoDistance[];
    /**
     * [DEPRECATED] HG5 based geographical keys related to listing address
     */
    geoHierarchy?: GeoHierarchy;
    /**
     * Keys that represent all the geographical areas/polygons that overlap with the location of
     * the listing
     */
    geoTags?: string[];
    /**
     * The name of the city or town
     */
    locality?: string;
    /**
     * The postal code. 4 digits if in Switzerland
     */
    postalCode?: string;
    /**
     * The post office box number
     */
    postOfficeBoxNumber?: string;
    /**
     * The canton code if in Switzerland. Otherwise the name of the region.
     */
    region?: string;
    /**
     * The street name and number
     */
    street?: string;
    /**
     * Additional street address information
     */
    streetAddition?: string;
    /**
     * [DEPRECATED] The street number
     */
    streetNumber?: string;
}
/**
 * The two letter country code
 */
export declare enum Country {
    Ad = "AD",
    Ae = "AE",
    Af = "AF",
    Ag = "AG",
    Ai = "AI",
    Al = "AL",
    Am = "AM",
    An = "AN",
    Ao = "AO",
    Aq = "AQ",
    Ar = "AR",
    As = "AS",
    At = "AT",
    Au = "AU",
    Aw = "AW",
    Ax = "AX",
    Az = "AZ",
    Ba = "BA",
    Bb = "BB",
    Bd = "BD",
    Be = "BE",
    Bf = "BF",
    Bg = "BG",
    Bh = "BH",
    Bi = "BI",
    Bj = "BJ",
    Bl = "BL",
    Bm = "BM",
    Bn = "BN",
    Bo = "BO",
    Bq = "BQ",
    Br = "BR",
    Bs = "BS",
    Bt = "BT",
    Bv = "BV",
    Bw = "BW",
    By = "BY",
    Bz = "BZ",
    Ca = "CA",
    Cc = "CC",
    Cd = "CD",
    Cf = "CF",
    Cg = "CG",
    Ch = "CH",
    Ci = "CI",
    Ck = "CK",
    Cl = "CL",
    Cm = "CM",
    Cn = "CN",
    Co = "CO",
    Cr = "CR",
    Cu = "CU",
    Cv = "CV",
    Cw = "CW",
    Cx = "CX",
    Cy = "CY",
    Cz = "CZ",
    De = "DE",
    Dj = "DJ",
    Dk = "DK",
    Dm = "DM",
    Do = "DO",
    Dz = "DZ",
    Ec = "EC",
    Ee = "EE",
    Eg = "EG",
    Eh = "EH",
    Er = "ER",
    Es = "ES",
    Et = "ET",
    Fi = "FI",
    Fj = "FJ",
    Fk = "FK",
    Fm = "FM",
    Fo = "FO",
    Fr = "FR",
    Ga = "GA",
    Gb = "GB",
    Gd = "GD",
    Ge = "GE",
    Gf = "GF",
    Gg = "GG",
    Gh = "GH",
    Gi = "GI",
    Gl = "GL",
    Gm = "GM",
    Gn = "GN",
    Gp = "GP",
    Gq = "GQ",
    Gr = "GR",
    Gs = "GS",
    Gt = "GT",
    Gu = "GU",
    Gw = "GW",
    Gy = "GY",
    Hk = "HK",
    Hm = "HM",
    Hn = "HN",
    Hr = "HR",
    Ht = "HT",
    Hu = "HU",
    Id = "ID",
    Ie = "IE",
    Il = "IL",
    Im = "IM",
    In = "IN",
    Io = "IO",
    Iq = "IQ",
    Ir = "IR",
    Is = "IS",
    It = "IT",
    Je = "JE",
    Jm = "JM",
    Jo = "JO",
    Jp = "JP",
    Ke = "KE",
    Kg = "KG",
    Kh = "KH",
    Ki = "KI",
    Km = "KM",
    Kn = "KN",
    Kp = "KP",
    Kr = "KR",
    Kw = "KW",
    Ky = "KY",
    Kz = "KZ",
    La = "LA",
    Lb = "LB",
    Lc = "LC",
    Li = "LI",
    Lk = "LK",
    Lr = "LR",
    Ls = "LS",
    Lt = "LT",
    Lu = "LU",
    Lv = "LV",
    Ly = "LY",
    Ma = "MA",
    Mc = "MC",
    Md = "MD",
    Me = "ME",
    Mf = "MF",
    Mg = "MG",
    Mh = "MH",
    Mk = "MK",
    Ml = "ML",
    Mm = "MM",
    Mn = "MN",
    Mo = "MO",
    Mp = "MP",
    Mq = "MQ",
    Mr = "MR",
    Ms = "MS",
    Mt = "MT",
    Mu = "MU",
    Mv = "MV",
    Mw = "MW",
    Mx = "MX",
    My = "MY",
    Mz = "MZ",
    Na = "NA",
    Nc = "NC",
    Ne = "NE",
    Nf = "NF",
    Ng = "NG",
    Ni = "NI",
    Nl = "NL",
    No = "NO",
    Np = "NP",
    Nr = "NR",
    Nu = "NU",
    Nz = "NZ",
    Om = "OM",
    Oo = "OO",
    Pa = "PA",
    Pe = "PE",
    Pf = "PF",
    Pg = "PG",
    Ph = "PH",
    Pk = "PK",
    Pl = "PL",
    Pm = "PM",
    Pn = "PN",
    Pr = "PR",
    Ps = "PS",
    Pt = "PT",
    Pw = "PW",
    Py = "PY",
    Qa = "QA",
    Re = "RE",
    Ro = "RO",
    Rs = "RS",
    Ru = "RU",
    Rw = "RW",
    Sa = "SA",
    Sb = "SB",
    Sc = "SC",
    Sd = "SD",
    Se = "SE",
    Sg = "SG",
    Sh = "SH",
    Si = "SI",
    Sj = "SJ",
    Sk = "SK",
    Sl = "SL",
    Sm = "SM",
    Sn = "SN",
    So = "SO",
    Sr = "SR",
    Ss = "SS",
    St = "ST",
    Sv = "SV",
    Sx = "SX",
    Sy = "SY",
    Sz = "SZ",
    Tc = "TC",
    Td = "TD",
    Tf = "TF",
    Tg = "TG",
    Th = "TH",
    Tj = "TJ",
    Tk = "TK",
    Tl = "TL",
    Tm = "TM",
    Tn = "TN",
    To = "TO",
    Tr = "TR",
    Tt = "TT",
    Tv = "TV",
    Tw = "TW",
    Tz = "TZ",
    Ua = "UA",
    Ug = "UG",
    Um = "UM",
    Us = "US",
    Uy = "UY",
    Uz = "UZ",
    Va = "VA",
    Vc = "VC",
    Ve = "VE",
    Vg = "VG",
    Vi = "VI",
    Vn = "VN",
    Vu = "VU",
    Wf = "WF",
    Ws = "WS",
    Ye = "YE",
    Yt = "YT",
    Za = "ZA",
    Zm = "ZM",
    Zw = "ZW"
}
/**
 * The geo coordinate representation of the listing address
 *
 * GPS coordinates represented in decimal degrees
 */
export declare class GeoCoordinates {
    /**
     * How accurate the coordinates are.  HIGH: House Match (Street with number) or Manual Set |
     * MEDIUM: Street Match | LOW: City | MIN: Lower Accuracy than City
     */
    accuracy?: Accuracy;
    /**
     * The height above or below sea-level in decimal degrees
     */
    elevation?: number;
    /**
     * The north-south position in decimal degrees
     */
    latitude?: number;
    /**
     * The east-west position in decimal degrees
     */
    longitude?: number;
}
/**
 * How accurate the coordinates are.  HIGH: House Match (Street with number) or Manual Set |
 * MEDIUM: Street Match | LOW: City | MIN: Lower Accuracy than City
 */
export declare enum Accuracy {
    High = "HIGH",
    Low = "LOW",
    Medium = "MEDIUM",
    Min = "MIN"
}
export declare class GeoDistance {
    /**
     * The distance (in meters) from the edge of the polygon to the listing coordinates.
     */
    distance?: number;
    /**
     * Geo tag for the polygon.
     */
    geoTag?: string;
}
/**
 * [DEPRECATED] HG5 based geographical keys related to listing address
 */
export declare class GeoHierarchy {
    /**
     * (LEGACY) The numeric id of the country as defined by HG5
     */
    countrySearchNameId?: number;
    /**
     * (LEGACY) The numeric id of the postal code and city as defined by HG5
     */
    placeId?: number;
    /**
     * (LEGACY) The numeric ids of the canton, regions, and country as defined by HG5
     */
    searchNameIds?: number[];
}
/**
 * Federal statistical office specific information
 */
export declare class BfsData {
    /**
     * The building identifier of the swiss population register
     */
    egid?: string;
    /**
     * The property (plot) identifier of the swiss population register
     */
    egrid?: string;
    /**
     * The flat identifier of the swiss population register
     */
    ewid?: string;
    /**
     * A list of flats with fields of the swiss population register in case ewid is not set but
     * multiple to match listing to
     */
    ewidCandidates?: EwidCandidate[];
}
export declare class EwidCandidate {
    /**
     * local flat number
     */
    ewid?: string;
    /**
     * the floor
     */
    floor?: number;
    /**
     * living space in square meters
     */
    livingSpace?: number;
    /**
     * no of rooms
     */
    rooms?: number;
}
export declare enum Category {
    AdvertisingArea = "ADVERTISING_AREA",
    AgriculturalInstallation = "AGRICULTURAL_INSTALLATION",
    AgriculturalLand = "AGRICULTURAL_LAND",
    AlottmentGarden = "ALOTTMENT_GARDEN",
    Apartment = "APARTMENT",
    Arcade = "ARCADE",
    Atelier = "ATELIER",
    Attic = "ATTIC",
    AtticCompartment = "ATTIC_COMPARTMENT",
    AtticFlat = "ATTIC_FLAT",
    BachelorFlat = "BACHELOR_FLAT",
    Bakery = "BAKERY",
    Bar = "BAR",
    BedAndBreakfast = "BED_AND_BREAKFAST",
    BifamiliarHouse = "BIFAMILIAR_HOUSE",
    BoatDryDock = "BOAT_DRY_DOCK",
    BoatLandingStage = "BOAT_LANDING_STAGE",
    BoatMooring = "BOAT_MOORING",
    BuildingLand = "BUILDING_LAND",
    Bungalow = "BUNGALOW",
    Butcher = "BUTCHER",
    CafeBar = "CAFE_BAR",
    Campground = "CAMPGROUND",
    CarPark = "CAR_PARK",
    CarpentryShop = "CARPENTRY_SHOP",
    Casino = "CASINO",
    Castle = "CASTLE",
    CaveHouse = "CAVE_HOUSE",
    CellarCompartment = "CELLAR_COMPARTMENT",
    Chalet = "CHALET",
    CheeseFactory = "CHEESE_FACTORY",
    ClubDisco = "CLUB_DISCO",
    Coffeehouse = "COFFEEHOUSE",
    Commercial = "COMMERCIAL",
    CommercialLand = "COMMERCIAL_LAND",
    CoveredParkingPlaceBike = "COVERED_PARKING_PLACE_BIKE",
    CoveredSlot = "COVERED_SLOT",
    DepartmentStore = "DEPARTMENT_STORE",
    DisplayWindow = "DISPLAY_WINDOW",
    DoubleGarage = "DOUBLE_GARAGE",
    Duplex = "DUPLEX",
    EngadineHouse = "ENGADINE_HOUSE",
    ExhibitionSpace = "EXHIBITION_SPACE",
    Factory = "FACTORY",
    Farm = "FARM",
    FarmHouse = "FARM_HOUSE",
    Flat = "FLAT",
    FuelStation = "FUEL_STATION",
    FurnishedFlat = "FURNISHED_FLAT",
    Garage = "GARAGE",
    GardenApartment = "GARDEN_APARTMENT",
    Gardening = "GARDENING",
    GolfCourse = "GOLF_COURSE",
    GrannyFlat = "GRANNY_FLAT",
    Hairdresser = "HAIRDRESSER",
    HobbyRoom = "HOBBY_ROOM",
    Home = "HOME",
    HorseBox = "HORSE_BOX",
    Hospital = "HOSPITAL",
    Hotel = "HOTEL",
    House = "HOUSE",
    HousePart = "HOUSE_PART",
    IndoorSwimmingPool = "INDOOR_SWIMMING_POOL",
    IndoorTennisCourt = "INDOOR_TENNIS_COURT",
    IndustrialLand = "INDUSTRIAL_LAND",
    IndustrialObject = "INDUSTRIAL_OBJECT",
    Institution = "INSTITUTION",
    Kiosk = "KIOSK",
    Laboratory = "LABORATORY",
    Library = "LIBRARY",
    Loft = "LOFT",
    Maisonette = "MAISONETTE",
    MiniGolfCourse = "MINI_GOLF_COURSE",
    Motel = "MOTEL",
    MountainFarm = "MOUNTAIN_FARM",
    MovieTheater = "MOVIE_THEATER",
    MultipleDwelling = "MULTIPLE_DWELLING",
    NursingHome = "NURSING_HOME",
    Office = "OFFICE",
    OldAgeHome = "OLD_AGE_HOME",
    OpenSlot = "OPEN_SLOT",
    OutdoorParkingPlaceBike = "OUTDOOR_PARKING_PLACE_BIKE",
    OutdoorSwimmingPool = "OUTDOOR_SWIMMING_POOL",
    ParkingSpace = "PARKING_SPACE",
    PartyRoom = "PARTY_ROOM",
    PatricianHouse = "PATRICIAN_HOUSE",
    Plot = "PLOT",
    Practice = "PRACTICE",
    Pub = "PUB",
    ResidentialCommercialBuilding = "RESIDENTIAL_COMMERCIAL_BUILDING",
    Restaurant = "RESTAURANT",
    Retail = "RETAIL",
    RetailSpace = "RETAIL_SPACE",
    RidingHall = "RIDING_HALL",
    RoofFlat = "ROOF_FLAT",
    RowHouse = "ROW_HOUSE",
    RusticHouse = "RUSTIC_HOUSE",
    Rustico = "RUSTICO",
    Sanatorium = "SANATORIUM",
    Sauna = "SAUNA",
    SharedApartment = "SHARED_APARTMENT",
    Shop = "SHOP",
    ShoppingCentre = "SHOPPING_CENTRE",
    SingleGarage = "SINGLE_GARAGE",
    SingleHouse = "SINGLE_HOUSE",
    SingleRoom = "SINGLE_ROOM",
    Solarium = "SOLARIUM",
    SportsHall = "SPORTS_HALL",
    SquashBadminton = "SQUASH_BADMINTON",
    StorageRoom = "STORAGE_ROOM",
    Studio = "STUDIO",
    TennisCourt = "TENNIS_COURT",
    TerraceFlat = "TERRACE_FLAT",
    TerraceHouse = "TERRACE_HOUSE",
    UndergroundSlot = "UNDERGROUND_SLOT",
    Villa = "VILLA",
    Warehouse = "WAREHOUSE",
    Workshop = "WORKSHOP"
}
/**
 * Set of features or attributes to better classify what the property can offer to the seeker
 */
export declare class Characteristics {
    areaSiaNf?: number;
    arePetsAllowed?: boolean;
    buildingFloorSize?: number;
    ceilingHeight?: number;
    craneCapacity?: number;
    cubage?: number;
    distanceHighSchool?: number;
    distanceKindergarten?: number;
    distanceMotorway?: number;
    distancePrimarySchool?: number;
    distancePublicTransport?: number;
    distanceShop?: number;
    elevatorCapacity?: number;
    floor?: number;
    floorLoad?: number;
    /**
     * Only for SALE listings: A gross premium is the total premium of an insurance contract
     * before brokerage or discounts have been deducted, in percent
     */
    grossPremium?: string;
    hallHeight?: number;
    hasAttic?: boolean;
    hasBalcony?: boolean;
    hasBuildingLawRestrictions?: boolean;
    hasCableTv?: boolean;
    hasCarPort?: boolean;
    hasCellar?: boolean;
    hasChargingStation?: boolean;
    hasConnectedBuildingLand?: boolean;
    hasDemolitionProperty?: boolean;
    hasDishwasher?: boolean;
    hasDoubleCarPort?: boolean;
    hasDoubleGarage?: boolean;
    hasElevator?: boolean;
    hasFireplace?: boolean;
    hasFlatSharingCommunity?: boolean;
    hasForeignQuota?: boolean;
    hasGarage?: boolean;
    hasGarageUnderground?: boolean;
    hasGardenShed?: boolean;
    hasGasSupply?: boolean;
    hasLakeView?: boolean;
    hasLiftingPlatform?: boolean;
    hasMountainView?: boolean;
    hasNiceView?: boolean;
    hasParking?: boolean;
    hasPhotovoltaic?: boolean;
    hasPlayground?: boolean;
    hasPowerSupply?: boolean;
    hasRamp?: boolean;
    hasRemoteViewings?: boolean;
    hasSewageSupply?: boolean;
    hasSteamer?: boolean;
    hasStoreRoom?: boolean;
    hasSwimmingPool?: boolean;
    hasThermalSolarCollector?: boolean;
    hasTiledStove?: boolean;
    hasTumbleDryer?: boolean;
    hasWashingMachine?: boolean;
    hasWaterSupply?: boolean;
    isChildFriendly?: boolean;
    isCornerHouse?: boolean;
    isDemolitionProperty?: boolean;
    isDilapidated?: boolean;
    isFirstOccupancy?: boolean;
    isGroundFloor?: boolean;
    isGroundFloorRaised?: boolean;
    isGutted?: boolean;
    isInNeedOfRenovation?: boolean;
    isInNeedOfRenovationPartially?: boolean;
    isLikeNew?: boolean;
    isMiddleHouse?: boolean;
    isMinergieCertified?: boolean;
    isMinergieGeneral?: boolean;
    isModernized?: boolean;
    isNewBuilding?: boolean;
    isOldBuilding?: boolean;
    isPartiallyRefurbished?: boolean;
    isProjection?: boolean;
    isQuiet?: boolean;
    isRefurbished?: boolean;
    isSecondaryResidenceAllowed?: boolean;
    isShellConstruction?: boolean;
    isSmokingAllowed?: boolean;
    isSunny?: boolean;
    isUnderRoof?: boolean;
    isWellTended?: boolean;
    isWheelchairAccessible?: boolean;
    livingSpace?: number;
    lotSize?: number;
    numberOfApartments?: number;
    numberOfBathrooms?: number;
    numberOfFloors?: number;
    numberOfParcels?: number;
    numberOfRooms?: number;
    numberOfShowers?: number;
    numberOfToilets?: number;
    numberOfToiletsGuest?: number;
    onEvenGround?: boolean;
    onHillside?: boolean;
    onHillsideSouth?: boolean;
    singleFloorSpace?: number;
    totalFloorSpace?: number;
    utilizationRatio?: number;
    utilizationRatioConstruction?: number;
    yearBuilt?: number;
    yearLastRenovated?: number;
}
/**
 * How the contact inquiry form should be configured on detail page
 */
export declare class ContactForm {
    deliveryFormat?: DeliveryFormat;
    /**
     * The number of fields present on the contact form
     */
    size?: Size;
}
export declare enum DeliveryFormat {
    CourtiersPartenaires = "COURTIERS_PARTENAIRES",
    Flowfact = "FLOWFACT",
    Normal = "NORMAL",
    NormalJson = "NORMAL_JSON",
    Rem = "REM"
}
/**
 * The number of fields present on the contact form
 */
export declare enum Size {
    Flowfact = "FLOWFACT",
    Maxi = "MAXI",
    Mini = "MINI",
    NoAddress = "NO_ADDRESS",
    WithAddress = "WITH_ADDRESS"
}
/**
 * The development state of the property.
 */
export declare enum Development {
    Full = "full",
    Partial = "partial",
    Undeveloped = "undeveloped"
}
export declare class ExternalIds {
    /**
     * The human-readable version of propertyReferenceId. To be used by seekers when talking to
     * CCC.
     */
    displayPropertyReferenceId?: string;
    /**
     * [DEPRECATED] See displayPropertyReferenceId
     */
    displayReferenceId?: string;
    /**
     * [DEPRECATED] See propertyReferenceId
     */
    internalReferenceId?: string;
    /**
     * Foreign key of listing that is provided by systems upstream of importer. If listing is
     * IDX-based, then it comes from third-party software. Usually the following format:
     * agency_id#ref_object#ref_house#ref_property.
     */
    propertyReferenceId?: string;
    /**
     * If listing is IDX-based, it usually contains ref_house. Alphanumeric.
     */
    refHouse?: string;
    /**
     * If listing is IDX-based, it comes usually contains ref_object. Alphanumeric.
     */
    refObject?: string;
    /**
     * If listing is IDX-based, it usually contains ref_property. Alphanumeric.
     */
    refProperty?: string;
    /**
     * The swissrets referenceId of the listing.
     * https://swissrets.ch/docs/noNamespace/complexType/exportType.properties.property.html#elem_referenceId
     */
    swissretsReferenceId?: string;
}
export declare class Legacy {
    /**
     * [DEPRECATED] Old name for listing id. Use id and meta.advertisementId instead
     */
    advertisementId?: number;
    /**
     * [DEPRECATED] Legacy database id in HG5 for the delivery_group table
     */
    deliveryGroupId?: number;
    /**
     * [DEPRECATED] Legacy database id in HG5 for the person table
     */
    personId?: number;
    /**
     * [DEPRECATED] The person id of the lister who submitted the listing. Only set if lister is
     * a PPA customer
     */
    ppaPersonId?: number;
    /**
     * [DEPRECATED] Legacy database id in HG5 for the publishing_group table
     */
    publishingGroupId?: number;
}
export declare class Lister {
    /**
     * The physical postal address of the lister
     */
    address?: Address;
    /**
     * Used to check if the seeker is allowed to contact the lister or not. e.g. !allowToContact
     * && valueAddedServices.isTenantPlusListing will prevent use from submitting the contact
     * form.
     */
    allowToContact?: boolean;
    /**
     * (Sensitive) The data relevant to billing and invoice systems
     */
    billing?: Billing;
    /**
     * Used to provide the seeker with information of how to get in contact
     */
    contacts?: Contacts;
    /**
     * The email of the lister
     */
    email?: string;
    /**
     * TODO investigate if still used
     */
    emailForRemFormat?: string;
    /**
     * The id of the lister specific to an external platform. Gets set to the origin lister-id
     * in case the id gets swapped to a pool id
     */
    externalPlatformId?: string;
    /**
     * The identifier of the agency. A pool agency is used if the listing is PPA
     */
    id?: string;
    /**
     * The legal entity name of the lister, if it is a company
     */
    legalName?: string;
    /**
     * The absolute url for the logo of the lister, if it is a company
     */
    logoUrl?: string;
    /**
     * The alternate, mobile phone number of the lister
     */
    mobile?: string;
    /**
     * The name of the lister
     */
    name?: string;
    /**
     * The default phone number of the lister
     */
    phone?: string;
    /**
     * The logon id of the lister on HG5 system
     */
    username?: string;
    /**
     * The website representing the lister. Typically if the lister is a company
     */
    website?: Website;
}
/**
 * (Sensitive) The data relevant to billing and invoice systems
 */
export declare class Billing {
    /**
     * The physical postal address used to notify lister of any billing or invoice information
     */
    address?: Address;
    /**
     * The company name as displayed on the invoice
     */
    companyName?: string;
    /**
     * The coupon code entered by user at the time of publishing the listing
     */
    couponId?: string;
    /**
     * The email used to notify lister of any billing or invoice information
     */
    email?: string;
    /**
     * The selected method delivery for the invoice
     */
    invoice?: Invoice;
    /**
     * The selected language for the invoice
     */
    language?: Language;
    /**
     * The legal name as displayed on the invoice
     */
    name?: string;
    /**
     * The day-time phone number used to notify lister of any billing or invoice information
     */
    phoneDay?: string;
    /**
     * The after hours phone number used to notify lister of any billing or invoice information
     */
    phoneEvening?: string;
    /**
     * The mobile phone number used to notify lister of any billing or invoice information
     */
    phoneMobile?: string;
    /**
     * The selected salutation as displayed on the invoice
     */
    salutation?: Salutation;
}
/**
 * The selected method delivery for the invoice
 */
export declare enum Invoice {
    Email = "EMAIL",
    Post = "POST",
    PostAndEmail = "POST_AND_EMAIL"
}
/**
 * The selected language for the invoice
 */
export declare enum Language {
    De = "DE",
    En = "EN",
    Fr = "FR",
    It = "IT"
}
/**
 * The selected salutation as displayed on the invoice
 */
export declare enum Salutation {
    Company = "COMPANY",
    Female = "FEMALE",
    Male = "MALE",
    Neutral = "NEUTRAL"
}
/**
 * Used to provide the seeker with information of how to get in contact
 */
export declare class Contacts {
    /**
     * General contact for all questions related to a listing
     */
    inquiry?: Person;
    /**
     * The person who will be present at the property viewing
     */
    viewing?: Person;
}
/**
 * General contact for all questions related to a listing
 *
 * The person who will be present at the property viewing
 */
export declare class Person {
    email?: string;
    /**
     * Last name or family name of the person
     */
    familyName?: string;
    /**
     * Short description or title of role of the person
     */
    function?: string;
    /**
     * Gender of the person
     */
    gender?: Gender;
    /**
     * First name of the person
     */
    givenName?: string;
    /**
     * Mobile phone number if the default is not reachable
     */
    mobile?: string;
    /**
     * Additional description of the person
     */
    note?: string;
    /**
     * Default phone number
     */
    phone?: string;
}
/**
 * Gender of the person
 */
export declare enum Gender {
    Female = "FEMALE",
    Male = "MALE",
    Other = "OTHER"
}
/**
 * The website representing the lister. Typically if the lister is a company
 */
export declare class Website {
    /**
     * The human-readable version of the link
     */
    label?: string;
    /**
     * The title of the link
     */
    title?: string;
    /**
     * The web link
     */
    value?: string;
}
/**
 * Any fields that are language-specific
 */
export declare class Localization {
    /**
     * The localized strings for the German locale. Currently only this one is used as the
     * default
     */
    de?: L10N;
    /**
     * The localized strings for the English locale. To be used once Homegate accepts swissRETS
     * listings
     */
    en?: L10N;
    /**
     * The localized strings for the French locale. To be used once Homegate accepts swissRETS
     * listings
     */
    fr?: L10N;
    /**
     * The localized strings for the Italian locale. To be used once Homegate accepts swissRETS
     * listings
     */
    it?: L10N;
    /**
     * The primary language of listing. Acts as a selector for the de,fr,it,en objects below.
     * Typically 'de'
     */
    primary?: string;
}
/**
 * The localized strings for the German locale. Currently only this one is used as the
 * default
 *
 * This object contains data fields that are language-specific
 *
 * The localized strings for the English locale. To be used once Homegate accepts swissRETS
 * listings
 *
 * The localized strings for the French locale. To be used once Homegate accepts swissRETS
 * listings
 *
 * The localized strings for the Italian locale. To be used once Homegate accepts swissRETS
 * listings
 */
export declare class L10N {
    attachments?: Attachment[];
    /**
     * Indicates whether the content has been machine translated
     */
    isMachineTranslated?: boolean;
    text?: Text;
    urls?: Url[];
}
export declare class Attachment {
    /**
     * Alternative text if file fails to load
     */
    alt?: string;
    /**
     * Caption of the file
     */
    caption?: null | string;
    /**
     * Brief description of the file
     */
    description?: null | string;
    /**
     * Filename without the URL path
     */
    file?: string;
    /**
     * [DEPRECATED]
     */
    publication?: null | string;
    /**
     * Title of the file
     */
    title?: string;
    /**
     * The type so the system can decide what to do with the file.
     */
    type?: AttachmentType;
    /**
     * Absolute or relative url to the file
     */
    url?: string;
}
/**
 * The type so the system can decide what to do with the file.
 */
export declare enum AttachmentType {
    Document = "DOCUMENT",
    Image = "IMAGE",
    Movie = "MOVIE",
    OfferLogo = "OFFER_LOGO",
    Plan = "PLAN",
    SalesBrochure = "SALES_BROCHURE"
}
export declare class Text {
    /**
     * A detailed description of the property. Can include basic html markup.
     */
    description?: string;
    /**
     * Description of where the property is located
     */
    situation?: string;
    /**
     * Brief summary or description of the listing
     */
    title?: string;
}
export declare class Url {
    label?: string;
    title?: string;
    type?: UrlType;
    value?: string;
}
export declare enum UrlType {
    Archilogic = "ARCHILOGIC",
    DirectLink = "DIRECT_LINK",
    Emonitor = "EMONITOR",
    Link = "LINK",
    VirtualTour = "VIRTUAL_TOUR",
    Youtube = "YOUTUBE"
}
export declare class Meta {
    /**
     * [DEPRECATED] Date and time when the export has been generated
     */
    created?: string;
    /**
     * Date and time when the importer received the listing creation request
     */
    createdAt?: string;
    /**
     * SemVer version of the HG-RETS schema this payload conforms to. Usually the latest schema
     * at the time the payload was generated
     */
    schemaVersion?: null | string;
    /**
     * The name of the software that generated the export
     */
    softwareName?: string;
    /**
     * Importer-upstream system that provided the listing data
     */
    source?: Source;
    /**
     * Date and time when the importer received the listing modification request
     */
    updatedAt?: string;
}
/**
 * Importer-upstream system that provided the listing data
 */
export declare enum Source {
    BusinessInsertionFunnel = "BUSINESS_INSERTION_FUNNEL",
    Filsinger = "FILSINGER",
    Flatfox = "FLATFOX",
    InsertionFunnel = "INSERTION_FUNNEL",
    SwissretsGateway = "SWISSRETS_GATEWAY",
    Unknown = "UNKNOWN"
}
/**
 * Data related to New Construction projects
 */
export declare class NewConstructionData {
    /**
     * The name of the new construction project
     */
    projectName?: string;
    /**
     * Type of new construction project
     */
    projectType?: ProjectType;
    /**
     * The URL of the new construction project
     */
    projectUrl?: string;
}
/**
 * Type of new construction project
 */
export declare enum ProjectType {
    NewConstructionLight = "NEW_CONSTRUCTION_LIGHT",
    NewConstructionPremium = "NEW_CONSTRUCTION_PREMIUM"
}
/**
 * Whether the listing is for sale or for rent
 */
export declare enum OfferType {
    Buy = "BUY",
    Rent = "RENT"
}
export declare class Prices {
    /**
     * Price structure used if the listing is for sale
     */
    buy?: Buy;
    /**
     * 3-letter currency code. See https://en.wikipedia.org/wiki/ISO_4217
     */
    currency?: string;
    /**
     * Price structure used if the listing is for rent
     */
    rent?: Rent;
}
/**
 * Price structure used if the listing is for sale
 */
export declare class Buy {
    area?: AreaThePriceRefersTo;
    /**
     * Extra costs not contained in the buying price
     */
    extra?: number;
    /**
     * The buying price
     */
    price?: number;
}
/**
 * How much the listing costs per unit of area. Usually used for plots of land.
 */
export declare enum AreaThePriceRefersTo {
    All = "ALL",
    Km2 = "KM2",
    M2 = "M2"
}
/**
 * Price structure used if the listing is for rent
 */
export declare class Rent {
    area?: AreaThePriceRefersTo;
    /**
     * Extra costs not contained in the rent price
     */
    extra?: number;
    /**
     * The total rent price, including extra costs. Typically the sum of net + extra
     */
    gross?: number;
    interval?: PriceInterval;
    /**
     * The rent price, not including extra costs
     */
    net?: number;
}
/**
 * How much the listing costs per unit of time. Typically monthly
 */
export declare enum PriceInterval {
    Day = "DAY",
    Month = "MONTH",
    Onetime = "ONETIME",
    Week = "WEEK",
    Year = "YEAR"
}
/**
 * Affected publisher by the custom fields
 */
export declare class Publisher {
    id?: string;
    /**
     * publisher specific attributes
     */
    options?: Option[];
}
export declare class Option {
    /**
     * ISO Datetime of expiration. 'Now' if omitted
     */
    expiration?: string;
    key?: string;
    lang?: string;
    /**
     * ISO Datetime of start. 'Now' if omitted
     */
    start?: string;
    value?: string;
}
/**
 * Any value added services that are associated with the listing
 */
export declare class ValueAddedServices {
    /**
     * The PPA Bundle ID of this listing
     */
    bundle?: string;
    /**
     * Used to determine when the listing is no longer to be exclusive
     */
    exclusiveUntil?: string;
    /**
     * Indicate if the lifetime toplisting is selected for this listing
     */
    isLifetimeToplisting?: boolean;
    /**
     * Should listing appear in showcase on front-page
     */
    isShowcase?: boolean;
    /**
     * Used to determine if it's a TenantPlus listing or not
     */
    isTenantPlusListing?: boolean;
    /**
     * Total number of listing images
     */
    numberOfImages?: number;
    /**
     * How listing ranking should be affected in search results. Premium and top listings are
     * ranked higher than the standard ones
     */
    rank?: Rank;
}
/**
 * How listing ranking should be affected in search results. Premium and top listings are
 * ranked higher than the standard ones
 */
export declare class Rank {
    /**
     * ISO Datetime of deactivation. 'Never' if omitted
     */
    end?: string;
    /**
     * ISO Datetime of activation. 'Now' if omitted
     */
    start?: string;
    type?: RankType;
}
export declare enum RankType {
    Basic = "BASIC",
    Premium = "PREMIUM",
    Standard = "STANDARD",
    Top = "TOP"
}
