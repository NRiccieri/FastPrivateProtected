mkdir fpp_code
touch fpp_code/__init__.py
cp -r ../../Clients ./fpp_code
cp -r ../../Model ./fpp_code
cp -r ../../Data ./fpp_code
docker image build . -t nriccieri/fpp_client:1.0
rm -rf fpp_code