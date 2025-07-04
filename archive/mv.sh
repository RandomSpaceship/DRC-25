for x in ./*.jpg; do
	new=$(echo "$x" | sed -e 's/:/-/g')
	mv "$x" "$new"
done

