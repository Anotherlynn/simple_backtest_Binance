import datetime

from _func.utility import *
from _func.downloadaggTrade import *

from zipfile import ZipFile
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
#
# gauth = GoogleAuth()
# drive = GoogleDrive(gauth)

YEARS = ['2018', '2019', '2020', '2021', '2022', '2023']
################################################################################################
folder_path = './data/spot/monthly/'  # Replace with the path to your folder

# Get a list of all the zip files in the folder
for folder in ['klines','aggTrades']:
    for symbol in ['ETHUSDT', 'ADAUSDT']:
        if folder == 'klines':
            zip_path = os.path.join(folder_path, folder, symbol, '1h/2018-01-01_2023-01-01/')
            names = ['Open time', 'Open', 'High', 'Low', 'Close','Volume', 'Close time', 'Quote asset volume',
                     'Number of trades', 'Taker buy base asset volume','Taker buy quote asset volume', 'Ignore']
        else:
            zip_path = os.path.join(folder_path, folder, symbol, '2018-01-01_2023-01-01/')
            names = ['Aggregate_tradeId','Price','Quantity','First_tradeId', 'Last_tradeId', 'Timestamp',
                     'Was_the_buyer_the_maker',	'Was_the_trade_the_best_price_match']
            # Append the DataFrame to the list
        # Loop through the zip files
        for y in YEARS:
            dfList = []
            for f in [f for f in os.listdir(zip_path) if f.endswith('.zip') and y in f]:
                # Open the zip file
                with ZipFile(os.path.join(zip_path, f)) as zf:
                    # Read the contents of› the zip file into a DataFrame
                    df_zip = pd.read_csv(zf.open(zf.namelist()[0]), index_col=False, header=None, names = names)
                    # Append the DataFrame to the list
                    dfList.append(df_zip)
            data = pd.concat(dfList, axis=0)
            outdir = "./data/"+y+"/"+symbol
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            data.to_csv(outdir+"/"+folder+".csv")

################################################################################################
# concat the kline.csv and aggTrade.csv according to requirements
# For trade data, merge them into kline data that use number of trade at a certain timestamp(trades_at_current_ts)
# and buy/sell ratio at a certain timestamp(buy_sell_ratio_at_current_ts)

# convert time function
def convert_time(unix):

    """
    convert Unix timestamp into GMT(+8) time
    :param unix:
    :return:
    """
    gmt = datetime.utcfromtimestamp(unix / 1000.0)
    formatted_timestamp = gmt.strftime('%Y-%m-%d %H')
    return formatted_timestamp


# The way to define buy-sell ratio is according to: https://dev.binance.vision/t/taker-buy-base-asset-volume/6026
for y in YEARS:
    for symbol in ['ETHUSDT', 'ADAUSDT']:
        kline = pd.read_csv("./data/"+y+"/"+symbol+"/klines.csv",index_col =0)
        agg = pd.read_csv("./data/"+y+"/"+symbol+"/aggTrades.csv",index_col =0)
        # generate new features

        kline['trades_at_current_tstrades_at_current_ts'] = kline['Taker buy base asset volume'] / (kline['Volume']- kline['Taker buy base asset volume'])
        kline['trades_at_current_ts'] = kline.groupby("Open time")['Number of trades'].transform('sum')
        # rename column to merge
        agg['Timestamp'] = [convert_time(i) for i in agg['Timestamp']]
        kline['Timestamp'] = [convert_time(i) for i in kline['Open time']]
        sss = pd.merge(kline, agg, how='inner', on=['Timestamp'])
        sss.to_csv("./data/"+y+"/binance_"+symbol[:3]+"_USDT_2018_2023.csv")
    print("%s data Done!" % y)
