typedef struct List_{
	int size;
	ListE head;
	ListE tail;
} List;

typedef struct ListE_{
	int secondary;
	double rating;
	int date;
	ListE_ *next;
} ListE;

void listStart(List *list);
void listEnd(List *list);
int listAddNext(List *list);

void listStart(List *list)
{
	list->size = 0;
	list->head = NULL;
	list->tail = NULL;
	return;
}

int listAddNext(List *list, int secondary, double rating, int date)
{
	ListE *newE = (ListE *)malloc(sizeof(ListE));
	newE->secondary = secondary;
	newE->rating = rating;
	newE->date = date;
	newE->next = NULL;
	if(list->size == 0)
		list->head = newE;
	else
		list->tail->next = newE;

	list->tail = newE;
// Free newE?
	list->size++;
	return 0;
}

int listDelNext(List *list)
{
	if(list->size == 0)
		return -1;

	ListE *oldE;
	oldE = list->head;
	list->head = oldE->next;
	free(oldE);
	return 0;
}

//////////////////////////////////////////////////////////////////

List movies[NUMMOVIES] = malloc(NUMMOVIES * sizeof(List));

for(a=0; a<NUMRATINGS; a++)
{
	i = dat[a].movie;
	u = dat[a].user;
	listAddNext(movies[i], u, dat[a].rating, dat[a].date);
}

List users[NUMUSERS] = malloc(NUMUSERS * sizeof(List));

for(a=0; a<NUMRATINGS; a++)
{
	i = dat[a].movie;
	u = dat[a].user;
	listAddNext(users[u], i, dat[a].rating, dat[a].date);
}